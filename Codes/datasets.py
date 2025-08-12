import os
import random
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

class MelSpectrogramTransform:
    """
    Converts an audio waveform to a log-mel spectrogram. Resamples if necessary.
    """
    def __init__(self, target_sample_rate=16000, n_mels=80, hop_length=160, win_length=400):
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, waveform):
        """Convert waveform to mel spectrogram"""
        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        # Convert to log-mel
        # log_mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec  # shape: [1, n_mels, T]

class SentencePieceTransform:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def text_to_ids(self, text):
        return self.sp.encode_as_ids(text.upper())

    def ids_to_text(self, ids):
        return self.sp.decode_ids(ids)

    def vocab_size(self):
        return self.sp.get_piece_size()

class MelSpectrogramSlidingDataset(Dataset):
    """
    For each audio file, we generate multiple (context, future) pairs by sliding over time
    with a given stride. We then return ALL pairs from that file in a list, which we flatten
    in the collate function.
    """
    def __init__(self, audio_files, context_frames, future_frames, mel_transform, stride=50):
        self.audio_files = audio_files
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.mel_transform = mel_transform
        self.stride = stride
        self.target_sample_rate = mel_transform.target_sample_rate

    def __len__(self):
        return len(self.audio_files)

    def load_and_transform(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
            
        # Apply mel transform (no need to pass sample_rate)
        mel = self.mel_transform(waveform)  # [1, n_mels, T]
        mel = mel.squeeze(0).transpose(0, 1)  # => [T, n_mels]
        return mel

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        mel_data = self.load_and_transform(audio_path)
        total_frames = mel_data.size(0)

        needed = self.context_frames + self.future_frames
        samples = []

        start_idx = 0
        while start_idx + needed <= total_frames:
            context = mel_data[start_idx : start_idx + self.context_frames]
            future = mel_data[start_idx + self.context_frames : start_idx + needed]

            # Exclude the chunk if 50% or more of the future data is zero
            if (future == 0).sum() >= future.numel() / 2:
                start_idx += self.stride
                continue

            samples.append((context, future, self.context_frames, self.future_frames))
            start_idx += self.stride

        if len(samples) == 0:
            return None
        return samples

class JointMultiTaskDataset(Dataset):
    """
    Dataset for multiple tasks with configurable task inputs and outputs
    """
    def __init__(self, task_configs, audio_transform, context_frames=240, future_frames=120):
        """
        Args:
            task_configs: Dictionary with task-specific configurations
                Each config should contain:
                - csv_path: Path to CSV file with data (optional for CPC)
                - tokenizer: Optional tokenizer for text data
                - task_type: Type of task ('asr', 'st', 'cpc', etc.)
            audio_transform: Transform to apply to audio data
            context_frames: Number of context frames for CPC task
            future_frames: Number of future frames for CPC task
        """
        # Store task configurations
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # Validate CSV files and ensure they have the same length
        csv_lengths = []
        self.task_dfs = {}
        
        for task_name, config in task_configs.items():
            if task_name != "CPC" and 'csv_path' in config:  # Skip CSV validation for CPC
                df = pd.read_csv(config['csv_path'])
                self.task_dfs[task_name] = df
                csv_lengths.append(len(df))
        
        if csv_lengths:
            if len(set(csv_lengths)) > 1:
                raise ValueError("All task CSV files must have the same length")
            self.length = csv_lengths[0]
        else:
            raise ValueError("At least one task must have a CSV file")
        
        # Store other parameters
        self.audio_transform = audio_transform
        self.context_frames = context_frames
        self.future_frames = future_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        result = {}
        
        # Process each task's data
        audio_path = None
        audio_features = None
        
        for task_name, config in self.task_configs.items():
            # Skip CPC in the main dataset - it has its own dataloader
            if task_name == "CPC":
                continue
                
            df = self.task_dfs[task_name]
            row = df.iloc[idx]
            
            # Get audio path (should be the first column in any task's dataframe)
            if audio_path is None:
                audio_path = row.iloc[0]
            
            # Load audio data if not already loaded
            if audio_features is None:
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                
                # Transform to mel spectrogram
                audio_features = self.audio_transform(waveform).squeeze(0).transpose(0, 1)  # [T, F]
                result['features'] = audio_features
            
            # Process text data if tokenizer is available
            if 'tokenizer' in config:
                text = row.iloc[1]  # Assuming text is the second column
                tokenizer = config['tokenizer']
                text_ids = tokenizer.text_to_ids(text)
                result[f'{task_name}_text'] = torch.tensor(text_ids)
        
        # For CPC task, split audio into context and future (for when CPC is used with ASR/ST data)
        total_frames = audio_features.shape[0]
        needed_frames = self.context_frames + self.future_frames
        
        # If audio is too short, pad
        if total_frames < needed_frames:
            padding = torch.zeros(needed_frames - total_frames, audio_features.shape[1])
            audio_features = torch.cat([audio_features, padding], dim=0)
            total_frames = needed_frames
        
        # Take a random segment for CPC
        if total_frames > needed_frames:
            start_idx = random.randint(0, total_frames - needed_frames)
            feat_segment = audio_features[start_idx:start_idx + needed_frames]
        else:
            feat_segment = audio_features
            
        result['context'] = feat_segment[:self.context_frames]
        result['future'] = feat_segment[self.context_frames:self.context_frames + self.future_frames]
        
        return result

def cpc_collate_fn(batch):
    """
    Each item in 'batch' is the list-of-chunks from a single audio file.
    We flatten all those chunks across *all* files in the batch, so we get a big
    dimension [N, context_frames, n_mels].
    """
    batch = [b for b in batch if b is not None]  # remove Nones
    if len(batch) == 0:
        return None

    all_contexts, all_futures, all_c_lens, all_f_lens = [], [], [], []
    for sample_list in batch:
        # sample_list is a list of (context, future, c_len, f_len)
        for (context, future, c_len, f_len) in sample_list:
            all_contexts.append(context)
            all_futures.append(future)
            all_c_lens.append(c_len)
            all_f_lens.append(f_len)

    contexts = torch.stack(all_contexts)  # => [N, context_frames, n_mels]
    futures = torch.stack(all_futures)    # => [N, future_frames, n_mels]
    c_lens = torch.tensor(all_c_lens, dtype=torch.long)
    f_lens = torch.tensor(all_f_lens, dtype=torch.long)

    return (contexts, futures, c_lens, f_lens)

def multitask_collate_fn(batch):
    """
    Collate function for multitask dataset
    """
    # Determine the keys from the first batch item
    first_item = batch[0]
    result = {}
    
    # Process audio features
    if 'features' in first_item:
        features = [item['features'] for item in batch]
        feat_lens = [f.shape[0] for f in features]
        result['features'] = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        result['feat_lens'] = torch.tensor(feat_lens)
    
    # Process text data for each task
    for key in first_item:
        if key.endswith('_text'):
            texts = [item[key] for item in batch]
            result[key] = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=1)
    
    # Process CPC data
    if 'context' in first_item and 'future' in first_item:
        contexts = [item['context'] for item in batch]
        futures = [item['future'] for item in batch]
        result['contexts'] = torch.stack(contexts)
        result['futures'] = torch.stack(futures)
    
    return result

def get_audio_files(data_dirs):
    """Get audio files from multiple directories"""
    audio_files = []
    for data_dir in data_dirs:
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith('.mp3'):
                    audio_files.append(os.path.join(root, f))
    
    random.shuffle(audio_files)
    print(f"Found {len(audio_files)} audio files.")
    return audio_files

def create_dataloaders(task_configs, audio_transform, batch_size=32, 
                      context_frames=240, future_frames=120, 
                      data_dirs=None, cpc_ratio=0.5):
    """
    Create dataloaders for training and validation
    
    Args:
        task_configs: Dictionary of task configurations
        audio_transform: Audio transform function
        batch_size: Batch size for dataloaders
        context_frames: Number of context frames for CPC
        future_frames: Number of future frames for CPC
        data_dirs: List of directories for audio files (for CPC)
        cpc_ratio: Ratio of CPC data in the batch
    
    Returns:
        Dictionary of dataloaders and tokenizers
    """
    # Check for tokenizers
    tokenizers = {}
    main_task_configs = {}
    
    # Split CPC from other tasks since it has its own dataloader
    for task_name, config in task_configs.items():
        if task_name == "CPC":
            # CPC has dedicated dataloader, no need to include in main dataset
            continue
            
        if 'tokenizer_path' in config:
            tokenizers[task_name] = SentencePieceTransform(config['tokenizer_path'])
            # Add tokenizer to config
            main_task_configs[task_name] = config.copy()
            main_task_configs[task_name]['tokenizer'] = tokenizers[task_name]
        else:
            main_task_configs[task_name] = config.copy()
    
    # Create main dataset for ASR/ST tasks (without CPC)
    joint_dataset = JointMultiTaskDataset(
        task_configs=main_task_configs,  # Use only non-CPC tasks
        audio_transform=audio_transform,
        context_frames=context_frames,
        future_frames=future_frames
    )
    
    # Split for train/test
    train_size = int(0.9 * len(joint_dataset))
    test_size = len(joint_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        joint_dataset, [train_size, test_size]
    )
    
    # Create CPC dataset if data_dirs is provided
    cpc_train_loader = None
    cpc_test_loader = None
    
    if "CPC" in task_configs and data_dirs is not None:
        # Apply transform specifically for CPC
        mel_transform = MelSpectrogramTransform(
            target_sample_rate=16000, n_mels=80, hop_length=160, win_length=400
        )
        
        # Get audio files
        audio_files = get_audio_files(data_dirs)
        
        # Split audio files for train/test
        train_files = audio_files[:int(0.9 * len(audio_files))]
        test_files = audio_files[int(0.9 * len(audio_files)):]
        
        # Create CPC datasets
        cpc_train_dataset = MelSpectrogramSlidingDataset(
            audio_files=train_files,
            context_frames=context_frames,
            future_frames=future_frames,
            mel_transform=mel_transform,
            stride=100
        )
        
        cpc_test_dataset = MelSpectrogramSlidingDataset(
            audio_files=test_files,
            context_frames=context_frames,
            future_frames=future_frames,
            mel_transform=mel_transform,
            stride=100
        )
        
        # Create CPC dataloaders
        cpc_train_loader = torch.utils.data.DataLoader(
            cpc_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=cpc_collate_fn
        )
        
        cpc_test_loader = torch.utils.data.DataLoader(
            cpc_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=True,
            collate_fn=cpc_collate_fn
        )
    
    # Create main dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size // 2,  # Half batch size for MoDo
        shuffle=True, 
        collate_fn=multitask_collate_fn,
        num_workers=4,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=multitask_collate_fn,
        num_workers=2
    )
    
    # Return dictionary with all loaders and tokenizers
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'cpc_train_loader': cpc_train_loader,
        'cpc_test_loader': cpc_test_loader,
        'tokenizers': tokenizers
    }