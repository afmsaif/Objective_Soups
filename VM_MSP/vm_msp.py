"""
VM-MSP: Vectorized Multilevel Multi-objective Speech Processing
Implementation for CoVoST v2 languages
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VMSPConfig:
    """Configuration for VM-MSP training"""
    # Model parameters
    encoder_dim: int = 512
    num_encoder_layers: int = 12
    num_attention_heads: int = 8
    
    # Training parameters
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    batch_size: int = 32
    num_epochs: int = 200
    
    # Languages and tasks
    languages: List[str] = None
    tasks: List[str] = None
    
    # Penalty parameters
    eta_init: float = 0.0
    eta1_init: float = 0.1
    eta_increase_rate: float = 0.02
    eta_max: float = 1.5
    
    # Optimization order
    optimization_order: str = "UAS"  # UAS or USA
    
    # Dynamic weighting
    use_dynamic_weighting: bool = True
    modo_lr: float = 0.01
    
    # Efficient training
    use_efficient_training: bool = True
    conflict_threshold: float = 0.0
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "fr", "de", "es", "ca"]
        if self.tasks is None:
            self.tasks = ["asr", "translation"]


class ConflictAvoidanceOptimizer:
    """
    Implements the MoDo algorithm for computing conflict-avoiding gradient directions
    """
    def __init__(self, num_objectives: int, lr: float = 0.01):
        self.num_objectives = num_objectives
        self.lr = lr
        self.lambda_weights = torch.ones(num_objectives) / num_objectives
        
    def compute_dynamic_weights(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute dynamic weights using MoDo algorithm
        
        Args:
            gradients: List of gradient tensors for each objective
        
        Returns:
            Dynamic weights for combining gradients
        """
        # Stack gradients
        grad_matrix = torch.stack([g.flatten() for g in gradients])
        
        # Compute gradient inner products
        grad_products = torch.mm(grad_matrix, grad_matrix.t())
        
        # Update lambda weights using projected gradient descent
        lambda_grad = grad_products @ self.lambda_weights
        self.lambda_weights = self.lambda_weights - self.lr * lambda_grad
        
        # Project onto simplex
        self.lambda_weights = self._project_simplex(self.lambda_weights)
        
        return self.lambda_weights
    
    def _project_simplex(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto probability simplex"""
        x_sorted, _ = torch.sort(x, descending=True)
        cumsum = torch.cumsum(x_sorted, dim=0)
        k = torch.arange(1, len(x) + 1, device=x.device, dtype=x.dtype)
        threshold = (cumsum - 1) / k
        indices = torch.where(x_sorted > threshold)[0]
        if len(indices) > 0:
            t = threshold[indices[-1]]
        else:
            t = threshold[-1]
        return torch.clamp(x - t, min=0)


class ConflictLayerDetector:
    """
    Detects layers with conflicting gradients for efficient training
    """
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.conflicting_layers = set()
        self.cosine_history = defaultdict(list)
        
    def detect_conflicts(self, model: nn.Module, task_gradients: Dict[str, Dict]) -> set:
        """
        Detect layers with conflicting gradients
        
        Args:
            model: The model being trained
            task_gradients: Dictionary of gradients for each task
        
        Returns:
            Set of layer indices with conflicts
        """
        conflicting_layers = set()
        
        for layer_idx, (name, param) in enumerate(model.named_parameters()):
            if "encoder" not in name:
                continue
                
            # Compute pairwise cosine similarities
            grad_list = []
            for task_key in task_gradients:
                if name in task_gradients[task_key]:
                    grad_list.append(task_gradients[task_key][name])
            
            if len(grad_list) < 2:
                continue
            
            # Calculate average cosine similarity
            similarities = []
            for i in range(len(grad_list)):
                for j in range(i + 1, len(grad_list)):
                    cos_sim = nn.functional.cosine_similarity(
                        grad_list[i].flatten().unsqueeze(0),
                        grad_list[j].flatten().unsqueeze(0)
                    ).item()
                    similarities.append(cos_sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            self.cosine_history[layer_idx].append(avg_similarity)
            
            if avg_similarity < self.threshold:
                conflicting_layers.add(layer_idx)
        
        self.conflicting_layers = conflicting_layers
        return conflicting_layers


class VMSPTrainer:
    """
    Main trainer for VM-MSP algorithm with optimization order selection
    """
    def __init__(self, model: nn.Module, config: VMSPConfig):
        self.model = model
        self.config = config
        
        # Initialize optimizers
        self.backbone_optimizer = optim.AdamW(
            [p for n, p in model.named_parameters() if 'head' not in n],
            lr=config.backbone_lr
        )
        self.head_optimizers = {}
        for lang in config.languages:
            for task in config.tasks:
                key = f"{lang}_{task}"
                self.head_optimizers[key] = optim.AdamW(
                    [p for n, p in model.named_parameters() if f'head_{key}' in n],
                    lr=config.head_lr
                )
        
        # Initialize conflict avoidance optimizer
        num_objectives = len(config.languages) * len(config.tasks) + 1  # +1 for self-supervised
        self.ca_optimizer = ConflictAvoidanceOptimizer(num_objectives, config.modo_lr)
        
        # Initialize conflict detector for efficient training
        if config.use_efficient_training:
            self.conflict_detector = ConflictLayerDetector(config.conflict_threshold)
        
        # Initialize penalty parameters
        self.eta = config.eta_init
        self.eta1 = config.eta1_init
        self.eta2 = config.eta_init
        
    def compute_losses(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for the batch
        
        Args:
            batch: Dictionary containing input data
        
        Returns:
            Dictionary of losses for each objective
        """
        losses = {}
        
        # Self-supervised loss (CPC)
        if 'unlabeled_audio' in batch:
            losses['self_supervised'] = self.compute_cpc_loss(batch['unlabeled_audio'])
        
        # Supervised losses for each language and task
        for lang in self.config.languages:
            if f'{lang}_audio' in batch:
                # ASR loss (CTC)
                if 'asr' in self.config.tasks:
                    losses[f'{lang}_asr'] = self.compute_ctc_loss(
                        batch[f'{lang}_audio'],
                        batch[f'{lang}_transcript']
                    )
                
                # Translation loss (Cross-entropy)
                if 'translation' in self.config.tasks:
                    losses[f'{lang}_translation'] = self.compute_ce_loss(
                        batch[f'{lang}_audio'],
                        batch[f'{lang}_translation']
                    )
        
        return losses
    
    def compute_cpc_loss(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute Contrastive Predictive Coding loss"""
        # Placeholder for CPC loss computation
        # In practice, this would involve:
        # 1. Encoding audio segments
        # 2. Predicting future segments
        # 3. Computing contrastive loss
        return torch.tensor(0.0, requires_grad=True)
    
    def compute_ctc_loss(self, audio: torch.Tensor, transcript: torch.Tensor) -> torch.Tensor:
        """Compute CTC loss for ASR"""
        # Placeholder for CTC loss computation
        return torch.tensor(0.0, requires_grad=True)
    
    def compute_ce_loss(self, audio: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for translation"""
        # Placeholder for CE loss computation
        return torch.tensor(0.0, requires_grad=True)
    
    def compute_gradients(self, losses: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """
        Compute gradients for each objective
        
        Args:
            losses: Dictionary of losses
        
        Returns:
            Dictionary of gradients for each objective
        """
        gradients = {}
        
        for task_key, loss in losses.items():
            self.backbone_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            gradients[task_key] = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[task_key][name] = param.grad.clone()
        
        return gradients
    
    def apply_multilevel_update(self, gradients: Dict[str, Dict], epoch: int):
        """
        Apply VM-MSP multilevel update based on optimization order
        
        Args:
            gradients: Dictionary of gradients for each objective
            epoch: Current epoch number
        """
        # Update penalty parameters
        self.update_penalty_parameters(epoch)
        
        # Get dynamic weights if enabled
        if self.config.use_dynamic_weighting:
            grad_list = []
            for task_key in gradients:
                if 'encoder.weight' in gradients[task_key]:
                    grad_list.append(gradients[task_key]['encoder.weight'])
            
            if grad_list:
                dynamic_weights = self.ca_optimizer.compute_dynamic_weights(grad_list)
            else:
                dynamic_weights = torch.ones(len(gradients)) / len(gradients)
        else:
            dynamic_weights = torch.ones(len(gradients)) / len(gradients)
        
        # Determine which layers to update (for efficient training)
        if self.config.use_efficient_training and epoch > 20:
            conflicting_layers = self.conflict_detector.detect_conflicts(
                self.model, gradients
            )
        else:
            conflicting_layers = None  # Update all layers
        
        # Apply updates based on optimization order
        if self.config.optimization_order == "UAS":
            self.apply_uas_update(gradients, dynamic_weights, conflicting_layers)
        elif self.config.optimization_order == "USA":
            self.apply_usa_update(gradients, dynamic_weights, conflicting_layers)
        else:
            raise ValueError(f"Unknown optimization order: {self.config.optimization_order}")
    
    def apply_uas_update(self, gradients: Dict, weights: torch.Tensor, 
                         conflicting_layers: Optional[set]):
        """
        Apply update with order: Unsupervised -> ASR -> Translation
        """
        self.backbone_optimizer.zero_grad()
        
        # Combine gradients according to VM-MSP formula
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if 'head' in name:
                continue
            
            # Skip non-conflicting layers if efficient training is enabled
            if conflicting_layers is not None and idx not in conflicting_layers:
                continue
            
            combined_grad = torch.zeros_like(param)
            weight_idx = 0
            
            # Level 1: ASR objectives
            for lang in self.config.languages:
                key = f'{lang}_asr'
                if key in gradients and name in gradients[key]:
                    combined_grad += weights[weight_idx] * gradients[key][name]
                    weight_idx += 1
            
            # Level 2: Translation objectives (with penalty η1)
            for lang in self.config.languages:
                key = f'{lang}_translation'
                if key in gradients and name in gradients[key]:
                    combined_grad += self.eta1 * weights[weight_idx] * gradients[key][name]
                    weight_idx += 1
            
            # Level 3: Self-supervised objective (with penalty η)
            if 'self_supervised' in gradients and name in gradients['self_supervised']:
                combined_grad += self.eta * gradients['self_supervised'][name]
            
            param.grad = combined_grad
        
        self.backbone_optimizer.step()
    
    def apply_usa_update(self, gradients: Dict, weights: torch.Tensor,
                         conflicting_layers: Optional[set]):
        """
        Apply update with order: Unsupervised -> Translation -> ASR
        """
        self.backbone_optimizer.zero_grad()
        
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if 'head' in name:
                continue
            
            # Skip non-conflicting layers if efficient training is enabled
            if conflicting_layers is not None and idx not in conflicting_layers:
                continue
            
            combined_grad = torch.zeros_like(param)
            weight_idx = 0
            
            # Level 1: Translation objectives
            for lang in self.config.languages:
                key = f'{lang}_translation'
                if key in gradients and name in gradients[key]:
                    combined_grad += weights[weight_idx] * gradients[key][name]
                    weight_idx += 1
            
            # Level 2: ASR objectives (with penalty η1)
            for lang in self.config.languages:
                key = f'{lang}_asr'
                if key in gradients and name in gradients[key]:
                    combined_grad += self.eta1 * weights[weight_idx] * gradients[key][name]
                    weight_idx += 1
            
            # Level 3: Self-supervised objective (with penalty η)
            if 'self_supervised' in gradients and name in gradients['self_supervised']:
                combined_grad += self.eta * gradients['self_supervised'][name]
            
            param.grad = combined_grad
        
        self.backbone_optimizer.step()
    
    def update_penalty_parameters(self, epoch: int):
        """Update penalty parameters based on epoch"""
        if epoch > 0:
            self.eta = min(self.eta + self.config.eta_increase_rate, self.config.eta_max)
            self.eta1 = min(self.eta1 + self.config.eta_increase_rate, self.config.eta_max)
            self.eta2 = min(self.eta2 + self.config.eta_increase_rate, self.config.eta_max)
            self.eta = self.eta1 * self.eta2  # Combined penalty for lowest level
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_losses = defaultdict(float)
        
        for batch_idx, batch in enumerate(dataloader):
            # Compute losses
            losses = self.compute_losses(batch)
            
            # Compute gradients
            gradients = self.compute_gradients(losses)
            
            # Apply multilevel update
            self.apply_multilevel_update(gradients, epoch)
            
            # Update task-specific heads
            for lang in self.config.languages:
                for task in self.config.tasks:
                    key = f"{lang}_{task}"
                    if key in losses:
                        self.head_optimizers[key].zero_grad()
                        losses[key].backward()
                        self.head_optimizers[key].step()
            
            # Track losses
            for key, loss in losses.items():
                total_losses[key] += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"η={self.eta:.3f}, η1={self.eta1:.3f}")
        
        # Log epoch statistics
        avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
        logger.info(f"Epoch {epoch} completed. Average losses: {avg_losses}")
        
        return avg_losses
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_wer = defaultdict(float)
        total_bleu = defaultdict(float)
        
        with torch.no_grad():
            for batch in dataloader:
                # Placeholder for actual evaluation
                # In practice, compute WER for ASR and BLEU for translation
                pass
        
        return total_wer, total_bleu


def create_model(config: VMSPConfig) -> nn.Module:
    """
    Create a model for VM-MSP training
    
    This is a placeholder - in practice, you would use a Conformer or Whisper model
    """
    class DummyModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.encoder = nn.Linear(80, config.encoder_dim)  # 80-dim log-mel features
            
            # Create task-specific heads
            for lang in config.languages:
                for task in config.tasks:
                    key = f"head_{lang}_{task}"
                    if task == "asr":
                        setattr(self, key, nn.Linear(config.encoder_dim, 1000))  # Vocab size
                    else:  # translation
                        setattr(self, key, nn.Linear(config.encoder_dim, 1000))
        
        def forward(self, x):
            return self.encoder(x)
    
    return DummyModel(config)


def main():
    """Main training script"""
    # Configuration
    config = VMSPConfig(
        languages=["en", "fr", "de", "es", "ca"],
        tasks=["asr", "translation"],
        optimization_order="UAS",  # Can be changed to "USA"
        use_dynamic_weighting=True,
        use_efficient_training=True,
        num_epochs=200
    )
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer = VMSPTrainer(model, config)
    
    # Placeholder for dataloader - in practice, load CoVoST v2 data
    train_dataloader = []  # Replace with actual dataloader
    val_dataloader = []    # Replace with actual dataloader
    
    # Training loop
    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        # avg_losses = trainer.train_epoch(train_dataloader, epoch)
        
        # Evaluate periodically
        if (epoch + 1) % 10 == 0:
            # wer, bleu = trainer.evaluate(val_dataloader)
            logger.info(f"Evaluation at epoch {epoch + 1}")
            # logger.info(f"WER: {wer}, BLEU: {bleu}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
