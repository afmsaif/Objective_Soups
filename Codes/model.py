import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import Conformer

class TaskHead(nn.Module):
    """Base class for task-specific heads"""
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def forward(self, x, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_parameters(self):
        """Get parameters specific to this task head"""
        return list(self.parameters())

class ASRHead(TaskHead):
    """Head for ASR task"""
    def __init__(self, encoder_dim, vocab_size, dropout=0.1):
        super().__init__("ASR")
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, vocab_size)
        )
    
    def forward(self, encoder_out, **kwargs):
        logits = self.head(encoder_out)
        return F.log_softmax(logits, dim=-1)

class TranslationHead(TaskHead):
    """Head for Speech Translation task"""
    def __init__(self, encoder_dim, vocab_size, dropout=0.1):
        super().__init__("ST")
        self.embedding = nn.Embedding(vocab_size, encoder_dim)
        self.decoder = nn.Transformer(
            d_model=encoder_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.output_projection = nn.Linear(encoder_dim, vocab_size)
    
    def forward(self, encoder_out, tgt_input=None, **kwargs):
        tgt_embedded = self.embedding(tgt_input)
        decoder_out = self.decoder(src=encoder_out, tgt=tgt_embedded)
        return F.log_softmax(self.output_projection(decoder_out), dim=-1)
    
    def get_parameters(self):
        """Get parameters specific to this task head"""
        return list(self.parameters())

class CPCHead(TaskHead):
    """Head for Contrastive Predictive Coding task"""
    def __init__(self, encoder_dim, future_frames=120):
        super().__init__("CPC")
        self.future_frames = future_frames
        self.predictor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.ReLU(),
            nn.Linear(encoder_dim * 2, encoder_dim)
        )
    
    def forward(self, encoder_out, **kwargs):
        # For CPC, we predict future frames from the last context frame
        context_enc = encoder_out  # [B, T, D]
        last_context = context_enc[:, -1, :]  # [B, D]
        
        # Predict each future step
        predicted_futures = []
        current_vec = last_context
        
        for _ in range(self.future_frames):
            pred = self.predictor(current_vec)  # [B, D]
            predicted_futures.append(pred)
            current_vec = pred  # Use prediction for next step
            
        predicted_future = torch.stack(predicted_futures, dim=1)  # [B, future_frames, D]
        return predicted_future

class MultiTaskModel(nn.Module):
    """Unified model with shared encoder and multiple task-specific heads"""
    def __init__(self, input_dim, encoder_dim, num_encoder_layers, heads_config=None):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        # Shared encoder for all tasks
        self.encoder = Conformer(
            num_classes=encoder_dim,
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_attention_heads=8,
            num_encoder_layers=num_encoder_layers,
            conv_kernel_size=31
        )
        
        # Initialize task heads
        self.task_heads = nn.ModuleDict()
        if heads_config:
            for head_config in heads_config:
                head_type = head_config.pop("type")
                
                if head_type == "ASR":
                    self.task_heads["ASR"] = ASRHead(encoder_dim=encoder_dim, **head_config)
                elif head_type == "ST":
                    self.task_heads["ST"] = TranslationHead(encoder_dim=encoder_dim, **head_config)
                elif head_type == "CPC":
                    self.task_heads["CPC"] = CPCHead(encoder_dim=encoder_dim, **head_config)
                else:
                    raise ValueError(f"Unknown head type: {head_type}")
    
    def forward(self, x, input_lengths, task=None, **kwargs):
        # Get encoder output
        encoder_out, out_len = self.encoder(x, input_lengths)
        
        # If task is specified, return output for that task only
        if task is not None:
            if task not in self.task_heads:
                raise ValueError(f"Task {task} not found. Available tasks: {list(self.task_heads.keys())}")
            
            # Call the specific task head
            if task == "ASR":
                return out_len, self.task_heads[task](encoder_out, **kwargs)
            else:
                return self.task_heads[task](encoder_out, **kwargs)
        
        # Otherwise, return all task outputs
        results = {}
        for task_name, task_head in self.task_heads.items():
            if task_name == "ASR":
                results[task_name] = (out_len, task_head(encoder_out, **kwargs))
            else:
                results[task_name] = task_head(encoder_out, **kwargs)
        
        return results
    
    def get_shared_parameters(self):
        """Get parameters of the shared encoder"""
        return list(self.encoder.parameters())
    
    def get_task_parameters(self, task_name):
        """Get parameters specific to a task head"""
        if task_name not in self.task_heads:
            raise ValueError(f"Task {task_name} not found. Available tasks: {list(self.task_heads.keys())}")
        return self.task_heads[task_name].get_parameters()
    
    def get_all_task_names(self):
        """Get names of all registered tasks"""
        return list(self.task_heads.keys())
    
    def add_task_head(self, head_type, **head_config):
        """Dynamically add a new task head"""
        if head_type == "ASR":
            self.task_heads["ASR"] = ASRHead(encoder_dim=self.encoder_dim, **head_config)
        elif head_type == "ST":
            self.task_heads["ST"] = TranslationHead(encoder_dim=self.encoder_dim, **head_config)
        elif head_type == "CPC":
            self.task_heads["CPC"] = CPCHead(encoder_dim=self.encoder_dim, **head_config)
        else:
            raise ValueError(f"Unknown head type: {head_type}")