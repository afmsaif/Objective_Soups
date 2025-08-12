import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

# Import custom modules
from model import MultiTaskModel
from datasets import create_dataloaders, MelSpectrogramTransform
from losses import MultiStepInfoNCE
from modo_utils import (
    set_seed, get_grad_vec, set_grad_vec, grad_modo, compute_task_gradients, 
    compute_pairwise_conflicts, analyze_layerwise_conflicts, 
    plot_conflict_heatmap, plot_top_conflicts_by_task_pair, 
    plot_lambda_evolution, plot_loss_curves
)

def train_epoch(model, 
               train_loader, 
               cpc_train_loader, 
               optimizer, 
               criterion_dict, 
               lambd, 
               task_configs, 
               device, 
               epoch, 
               modo_gamma=0.1, 
               modo_rho=0.01,
               output_dir="results_modo",
               analyze_conflicts=False):
    """
    Train for one epoch using MoDo
    
    Args:
        model: The model to train
        train_loader: Dataloader for training data
        cpc_train_loader: Dataloader for CPC data
        optimizer: Optimizer
        criterion_dict: Dictionary of loss functions for each task
        lambd: Lambda weights for MoDo
        task_configs: Dictionary of task configurations
        device: Device to train on
        epoch: Current epoch number
        modo_gamma: Learning rate for lambda update
        modo_rho: Regularization parameter
        output_dir: Directory to save results
        analyze_conflicts: Whether to analyze conflicts between tasks
    
    Returns:
        Updated lambda weights, average losses
    """
    model.train()
    running_losses = {task: 0.0 for task in task_configs.keys()}
    total_loss = 0.0
    epoch_lambda = []
    
    # Create iterators for the dataloaders
    data_iter = iter(train_loader)
    cpc_data_iter = iter(cpc_train_loader) if cpc_train_loader else None
    
    # Progress bar
    num_steps = len(train_loader)
    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch}")
    
    # Get task names in a fixed order
    task_names = list(task_configs.keys())
    
    # Get shared parameters (encoder)
    if isinstance(model, nn.DataParallel):
        shared_params = model.module.get_shared_parameters()
    else:
        shared_params = model.get_shared_parameters()
    
    # Convert shared_params to a set of parameter IDs for efficient lookup
    shared_param_ids = {id(p) for p in shared_params}
    
    # Store first batch conflicts for analysis
    first_batch_conflicts = None
    first_batch_layer_conflicts = None
    
    for batch_idx in pbar:
        # Get two batches for MoDo
        try:
            batch1 = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch1 = next(data_iter)
            
        try:
            batch2 = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch2 = next(data_iter)
        
        # Get CPC batches if needed
        cpc_batch1 = None
        cpc_batch2 = None
        if "CPC" in task_configs and cpc_data_iter:
            try:
                cpc_batch1 = next(cpc_data_iter)
            except StopIteration:
                cpc_data_iter = iter(cpc_train_loader)
                cpc_batch1 = next(cpc_data_iter)
                
            try:
                cpc_batch2 = next(cpc_data_iter)
            except StopIteration:
                cpc_data_iter = iter(cpc_train_loader)
                cpc_batch2 = next(cpc_data_iter)
                
            # Skip if any batch is None (can happen with CPC data)
            if cpc_batch1 is None or cpc_batch2 is None:
                continue
        
        # Compute gradients for shared parameters (encoder) using MoDo
        # Batch 1
        grad_list1 = []
        losses1 = []
        
        # For each task, compute gradients for the shared parameters
        for task_idx, task_name in enumerate(task_names):
            optimizer.zero_grad()
            
            if task_name == "CPC" and cpc_data_iter and cpc_batch1 is not None:
                # Use dedicated CPC dataloader
                contexts, futures, c_lens, f_lens = cpc_batch1
                contexts = contexts.to(device)
                futures = futures.to(device)
                c_lens = c_lens.to(device)
                f_lens = f_lens.to(device)
                
                # Forward pass for CPC
                predicted_future = model(x=contexts, input_lengths=c_lens, task="CPC")
                
                # Encode target frames
                if isinstance(model, nn.DataParallel):
                    future_enc, _ = model.module.encoder(futures, f_lens)
                else:
                    future_enc, _ = model.encoder(futures, f_lens)
                    
                # Compute loss
                loss = criterion_dict["CPC"](predicted_future, future_enc)
            else:
                # Regular tasks from main dataloader
                # Skip if this task is not in the batch
                if f'{task_name}_text' not in batch1:
                    continue
                    
                # Forward pass
                if task_name == "ASR":
                    features = batch1['features'].to(device)
                    feature_lengths = batch1['feat_lens'].to(device)
                    asr_targets = batch1[f'ASR_text'].to(device)
                    
                    out_len, asr_logits = model(x=features, input_lengths=feature_lengths, task="ASR")
                    asr_logits = asr_logits.transpose(0, 1)  # (T, N, C)
                    target_lengths = torch.sum(asr_targets != 1, dim=1)
                    loss = criterion_dict["ASR"](asr_logits, asr_targets, out_len, target_lengths)
                    
                elif task_name == "ST":
                    features = batch1['features'].to(device)
                    feature_lengths = batch1['feat_lens'].to(device)
                    trans_targets = batch1[f'ST_text'].to(device)
                    
                    tgt_input = trans_targets[:, :-1]
                    tgt_output = trans_targets[:, 1:]
                    st_logits = model(x=features, input_lengths=feature_lengths, tgt_input=tgt_input, task="ST")
                    loss = criterion_dict["ST"](st_logits.view(-1, st_logits.size(-1)), tgt_output.reshape(-1))
                else:
                    # For any other task, follow a similar pattern
                    # This is a placeholder for future task types
                    continue
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Get gradients for shared parameters only
            shared_grad = get_grad_vec(model, shared_params)
            grad_list1.append(shared_grad)
            losses1.append(loss.item())
            
            # Reset gradients
            optimizer.zero_grad()
            
        # Repeat for batch 2
        grad_list2 = []
        losses2 = []
        
        for task_idx, task_name in enumerate(task_names):
            optimizer.zero_grad()
            
            if task_name == "CPC" and cpc_data_iter and cpc_batch2 is not None:
                # Use dedicated CPC dataloader
                contexts, futures, c_lens, f_lens = cpc_batch2
                contexts = contexts.to(device)
                futures = futures.to(device)
                c_lens = c_lens.to(device)
                f_lens = f_lens.to(device)
                
                # Forward pass for CPC
                predicted_future = model(x=contexts, input_lengths=c_lens, task="CPC")
                
                # Encode target frames
                if isinstance(model, nn.DataParallel):
                    future_enc, _ = model.module.encoder(futures, f_lens)
                else:
                    future_enc, _ = model.encoder(futures, f_lens)
                    
                # Compute loss
                loss = criterion_dict["CPC"](predicted_future, future_enc)
            else:
                # Regular tasks from main dataloader
                # Skip if this task is not in the batch
                if f'{task_name}_text' not in batch2:
                    continue
                    
                # Forward pass
                if task_name == "ASR":
                    features = batch2['features'].to(device)
                    feature_lengths = batch2['feat_lens'].to(device)
                    asr_targets = batch2[f'ASR_text'].to(device)
                    
                    out_len, asr_logits = model(x=features, input_lengths=feature_lengths, task="ASR")
                    asr_logits = asr_logits.transpose(0, 1)  # (T, N, C)
                    target_lengths = torch.sum(asr_targets != 1, dim=1)
                    loss = criterion_dict["ASR"](asr_logits, asr_targets, out_len, target_lengths)
                    
                elif task_name == "ST":
                    features = batch2['features'].to(device)
                    feature_lengths = batch2['feat_lens'].to(device)
                    trans_targets = batch2[f'ST_text'].to(device)
                    
                    tgt_input = trans_targets[:, :-1]
                    tgt_output = trans_targets[:, 1:]
                    st_logits = model(x=features, input_lengths=feature_lengths, tgt_input=tgt_input, task="ST")
                    loss = criterion_dict["ST"](st_logits.view(-1, st_logits.size(-1)), tgt_output.reshape(-1))
                else:
                    # For any other task, follow a similar pattern
                    continue
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Get gradients for shared parameters only
            shared_grad = get_grad_vec(model, shared_params)
            grad_list2.append(shared_grad)
            losses2.append(loss.item())
            
            # Reset gradients
            optimizer.zero_grad()
        
        # Check if we have enough tasks with gradients to proceed
        if len(grad_list1) < 1 or len(grad_list2) < 1:
            continue
        
        # Store first batch conflicts for analysis
        if batch_idx == 0 and analyze_conflicts and (epoch == 1 or epoch % 5 == 0):
            # Convert lists to tensors
            grads1 = torch.stack(grad_list1)
            
            # Compute conflicts between all task pairs
            all_conflicts = compute_pairwise_conflicts(grads1)
            
            # Analyze layerwise conflicts
            layer_conflicts = analyze_layerwise_conflicts(model, grads1, task_names)
            
            # Store for plotting later
            first_batch_conflicts = all_conflicts
            first_batch_layer_conflicts = layer_conflicts
        
        # Stack gradients into tensors
        grads1 = torch.stack(grad_list1)
        grads2 = torch.stack(grad_list2)
        
        # Apply MoDo to get update direction for shared parameters
        grad_list = [grads1, grads2]
        lambd_new, multi_grad = grad_modo(
            grad_list, lambd, gamma=modo_gamma, rho=modo_rho
        )
        
        # Update shared parameters (encoder) with multi-gradient
        optimizer.zero_grad()
        set_grad_vec(model, multi_grad, shared_params)
        
        # Step optimizer to update shared parameters
        optimizer.step()
        
        # Now update task-specific parameters one by one with their own gradients
        for task_idx, task_name in enumerate(task_names):
            # Get task-specific parameters
            if isinstance(model, nn.DataParallel):
                task_params = model.module.get_task_parameters(task_name)
            else:
                task_params = model.get_task_parameters(task_name)
                
            # Skip if there are no task-specific parameters
            if not task_params:
                continue
                
            optimizer.zero_grad()
            
            # Forward and backward pass to compute gradients for task-specific parameters
            if task_name == "CPC" and cpc_data_iter and cpc_batch1 is not None:
                contexts, futures, c_lens, f_lens = cpc_batch1  # Use batch1 for simplicity
                contexts = contexts.to(device)
                futures = futures.to(device)
                c_lens = c_lens.to(device)
                f_lens = f_lens.to(device)
                
                # Forward pass for CPC
                predicted_future = model(x=contexts, input_lengths=c_lens, task="CPC")
                
                # Encode target frames
                if isinstance(model, nn.DataParallel):
                    future_enc, _ = model.module.encoder(futures, f_lens)
                else:
                    future_enc, _ = model.encoder(futures, f_lens)
                    
                # Compute loss
                loss = criterion_dict["CPC"](predicted_future, future_enc)
            else:
                # Regular tasks from main dataloader
                # Skip if this task is not in the batch
                if f'{task_name}_text' not in batch1:
                    continue
                    
                if task_name == "ASR":
                    features = batch1['features'].to(device)
                    feature_lengths = batch1['feat_lens'].to(device)
                    asr_targets = batch1[f'ASR_text'].to(device)
                    
                    out_len, asr_logits = model(x=features, input_lengths=feature_lengths, task="ASR")
                    asr_logits = asr_logits.transpose(0, 1)  # (T, N, C)
                    target_lengths = torch.sum(asr_targets != 1, dim=1)
                    loss = criterion_dict["ASR"](asr_logits, asr_targets, out_len, target_lengths)
                    
                elif task_name == "ST":
                    features = batch1['features'].to(device)
                    feature_lengths = batch1['feat_lens'].to(device)
                    trans_targets = batch1[f'ST_text'].to(device)
                    
                    tgt_input = trans_targets[:, :-1]
                    tgt_output = trans_targets[:, 1:]
                    st_logits = model(x=features, input_lengths=feature_lengths, tgt_input=tgt_input, task="ST")
                    loss = criterion_dict["ST"](st_logits.view(-1, st_logits.size(-1)), tgt_output.reshape(-1))
                else:
                    # For any other task, follow a similar pattern
                    continue
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Only keep gradients for task-specific parameters and zero out shared params
            # Use parameter IDs for comparison to avoid tensor shape issues
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Zero out gradients for shared parameters (encoder)
                    if id(param) in shared_param_ids:
                        param.grad.zero_()
            
            # Step optimizer to update task-specific parameters
            optimizer.step()
            optimizer.zero_grad()
        
        # Update lambda for next iteration
        if len(lambd_new) == len(lambd):  # Ensure dimensions match
            lambd = lambd_new
            epoch_lambda.append(lambd.detach().cpu().numpy())
        
        # Update metrics
        for task_idx, task_name in enumerate(task_names):
            if task_idx < len(losses1):
                running_losses[task_name] += losses1[task_idx]
        total_loss += sum(losses1)
        
        # Update progress bar
        lambda_str = ", ".join([f"{l:.2f}" for l in lambd.tolist()])
        loss_str = ", ".join([f"{task}: {running_losses[task]/(batch_idx+1):.4f}" for task in task_names])
        pbar.set_postfix_str(f"Loss: {loss_str}, λ: [{lambda_str}]")
    
    # Compute average losses
    avg_losses = {task: losses / max(1, num_steps) for task, losses in running_losses.items()}
    
    # Plot conflicts if needed
    if analyze_conflicts and first_batch_conflicts and (epoch == 1 or epoch % 5 == 0):
        plot_conflict_heatmap(first_batch_conflicts, epoch, output_dir=output_dir, task_names=task_names)
        plot_top_conflicts_by_task_pair(first_batch_layer_conflicts, epoch, output_dir=output_dir, task_names=task_names)
    
    return lambd, np.mean(epoch_lambda, axis=0) if epoch_lambda else np.array([1.0/len(task_names)] * len(task_names)), avg_losses

def evaluate(model, test_loader, cpc_test_loader, criterion_dict, task_configs, device):
    """
    Evaluate model on test data
    
    Args:
        model: The model to evaluate
        test_loader: Dataloader for test data
        cpc_test_loader: Dataloader for CPC test data
        criterion_dict: Dictionary of loss functions for each task
        task_configs: Dictionary of task configurations
        device: Device to evaluate on
    
    Returns:
        Dictionary of average losses for each task
    """
    model.eval()
    test_losses = {task: 0.0 for task in task_configs.keys()}
    test_counts = {task: 0 for task in task_configs.keys()}
    
    # Get task names in a fixed order
    task_names = list(task_configs.keys())
    
    with torch.no_grad():
        # First evaluate on main test loader
        for batch in test_loader:
            # Process each task
            for task_name in task_names:
                if task_name == "CPC":
                    # Skip CPC task for now, we'll evaluate it on the dedicated dataloader
                    continue
                
                # Check if this task's data is in the batch
                if f'{task_name}_text' not in batch:
                    continue
                
                # Forward pass for this task
                if task_name == "ASR":
                    features = batch['features'].to(device)
                    feature_lengths = batch['feat_lens'].to(device)
                    asr_targets = batch[f'ASR_text'].to(device)
                    
                    out_len, asr_logits = model(x=features, input_lengths=feature_lengths, task="ASR")
                    asr_logits = asr_logits.transpose(0, 1)  # (T, N, C)
                    target_lengths = torch.sum(asr_targets != 1, dim=1)
                    loss = criterion_dict["ASR"](asr_logits, asr_targets, out_len, target_lengths)
                    
                elif task_name == "ST":
                    features = batch['features'].to(device)
                    feature_lengths = batch['feat_lens'].to(device)
                    trans_targets = batch[f'ST_text'].to(device)
                    
                    tgt_input = trans_targets[:, :-1]
                    tgt_output = trans_targets[:, 1:]
                    st_logits = model(x=features, input_lengths=feature_lengths, tgt_input=tgt_input, task="ST")
                    loss = criterion_dict["ST"](st_logits.view(-1, st_logits.size(-1)), tgt_output.reshape(-1))
                else:
                    # For any other task, follow a similar pattern
                    continue
                
                # Update test losses
                test_losses[task_name] += loss.item()
                test_counts[task_name] += 1
        
        # Then evaluate CPC on dedicated dataloader if available
        if "CPC" in task_configs and cpc_test_loader:
            cpc_count = 0
            for batch in cpc_test_loader:
                if batch is None:
                    continue
                
                cpc_count += 1
                contexts, futures, c_lens, f_lens = batch
                contexts = contexts.to(device)
                futures = futures.to(device)
                c_lens = c_lens.to(device)
                f_lens = f_lens.to(device)
                
                # Forward pass for CPC
                predicted_future = model(x=contexts, input_lengths=c_lens, task="CPC")
                
                # Encode target frames
                if isinstance(model, nn.DataParallel):
                    future_enc, _ = model.module.encoder(futures, f_lens)
                else:
                    future_enc, _ = model.encoder(futures, f_lens)
                    
                # Compute loss
                loss = criterion_dict["CPC"](predicted_future, future_enc)
                
                # Update CPC loss
                test_losses["CPC"] += loss.item()
                test_counts["CPC"] += 1
    
    # Normalize losses by the number of batches
    avg_test_losses = {}
    for task_name in task_configs.keys():
        if test_counts[task_name] > 0:
            avg_test_losses[task_name] = test_losses[task_name] / test_counts[task_name]
        else:
            avg_test_losses[task_name] = float('inf')  # Indicate that this task wasn't evaluated
    
    return avg_test_losses

def main(args):
    # Set up directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Define audio transform
    audio_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, hop_length=160, n_fft=512, win_length=400
    )
    
    # Define task configurations
    task_configs = {}
    
    # Add ASR task if requested
    if hasattr(args, 'asr_csv_path') and args.asr_csv_path:
        task_configs["ASR"] = {
            "tokenizer_path": args.asr_tokenizer_path,
            "csv_path": args.asr_csv_path,
        }
    
    # Add ST task if requested
    if hasattr(args, 'st_csv_path') and args.st_csv_path:
        task_configs["ST"] = {
            "tokenizer_path": args.st_tokenizer_path,
            "csv_path": args.st_csv_path,
        }
    
    # Add CPC task if requested
    if hasattr(args, 'use_cpc') and args.use_cpc:
        task_configs["CPC"] = {
            "future_frames": args.future_frames,
        }
    
    # Check if we have any tasks
    if not task_configs:
        raise ValueError("No tasks specified! Please provide at least one task.")
    
    # Create dataloaders
    data = create_dataloaders(
        task_configs=task_configs,
        audio_transform=audio_transform,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        future_frames=args.future_frames,
        data_dirs=args.cpc_data_dirs if hasattr(args, 'cpc_data_dirs') and args.use_cpc else None,
    )
    
    train_loader = data['train_loader']
    test_loader = data['test_loader']
    cpc_train_loader = data['cpc_train_loader'] if args.use_cpc else None
    cpc_test_loader = data['cpc_test_loader'] if args.use_cpc else None
    tokenizers = data['tokenizers']
    
    # Update task configurations with vocab sizes
    for task, tok in tokenizers.items():
        task_configs[task]["vocab_size"] = tok.vocab_size()
    
    # Create model heads config
    heads_config = []
    
    # Add ASR head
    if "ASR" in task_configs:
        heads_config.append({
            "type": "ASR",
            "vocab_size": task_configs["ASR"]["vocab_size"],
            "dropout": args.dropout,
        })
    
    # Add ST head
    if "ST" in task_configs:
        heads_config.append({
            "type": "ST",
            "vocab_size": task_configs["ST"]["vocab_size"],
            "dropout": args.dropout,
        })
    
    # Add CPC head
    if "CPC" in task_configs:
        heads_config.append({
            "type": "CPC",
            "future_frames": args.future_frames,
        })
    
    # Initialize model
    model = MultiTaskModel(
        input_dim=80,  # 80 mel filterbanks
        encoder_dim=args.encoder_dim,
        num_encoder_layers=args.num_encoder_layers,
        heads_config=heads_config,
    ).to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Define loss functions
    criterion_dict = {}
    
    if "ASR" in task_configs:
        criterion_dict["ASR"] = nn.CTCLoss(blank=0, zero_infinity=True)
    
    if "ST" in task_configs:
        criterion_dict["ST"] = nn.CrossEntropyLoss(ignore_index=1)
    
    if "CPC" in task_configs:
        criterion_dict["CPC"] = MultiStepInfoNCE(temperature=0.1)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Initialize lambda weights (equal weight for each task)
    num_tasks = len(task_configs)
    lambd = torch.ones(num_tasks, device=device) / num_tasks
    
    # Track metrics
    train_losses = {task: [] for task in task_configs.keys()}
    test_losses = {task: [] for task in task_configs.keys()}
    lambda_history = []
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"{args.output_dir}/logs")
    
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        # Train
        lambd, epoch_lambda, avg_train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            cpc_train_loader=cpc_train_loader,
            optimizer=optimizer,
            criterion_dict=criterion_dict,
            lambd=lambd,
            task_configs=task_configs,
            device=device,
            epoch=epoch,
            modo_gamma=args.modo_gamma,
            modo_rho=args.modo_rho,
            output_dir=args.output_dir,
            analyze_conflicts=args.analyze_conflicts
        )
        
        # Store lambda history
        lambda_history.append(epoch_lambda)
        
        # Update train losses
        for task, loss in avg_train_losses.items():
            train_losses[task].append(loss)
            writer.add_scalar(f'Loss/train_{task}', loss, epoch)
        
        # Evaluate
        avg_test_losses = evaluate(
            model=model,
            test_loader=test_loader,
            cpc_test_loader=cpc_test_loader,
            criterion_dict=criterion_dict,
            task_configs=task_configs,
            device=device
        )
        
        # Update test losses
        for task, loss in avg_test_losses.items():
            if loss != float('inf'):  # Skip tasks that weren't evaluated
                test_losses[task].append(loss)
                writer.add_scalar(f'Loss/test_{task}', loss, epoch)
        
        # Log lambda values
        for i, task in enumerate(task_configs.keys()):
            if i < len(lambd):
                writer.add_scalar(f'Lambda/{task}', lambd[i].item(), epoch)
        
        # Print progress
        print(f"Epoch {epoch}/{args.num_epochs}")
        for task in task_configs.keys():
            train_loss = avg_train_losses.get(task, float('nan'))
            test_loss = avg_test_losses.get(task, float('nan'))
            if test_loss == float('inf'):
                test_loss = float('nan')  # For prettier printing
            print(f"  {task} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        lambda_values = [f"{x:.2f}" for x in lambd.tolist()]
        print(f"  Lambda: [{', '.join(lambda_values)}]")
        
        # Save model checkpoint
        if epoch % args.save_interval == 0 or epoch == args.num_epochs:
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            ckpt_path = os.path.join(args.output_dir, "models", f"multitask_modo_epoch_{epoch}.pth")
            torch.save(model_to_save.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
    
    # Plot final results if we have any lambda history
    if lambda_history:
        plot_lambda_evolution(
            lambda_history=np.array(lambda_history),
            output_dir=args.output_dir,
            task_names=list(task_configs.keys())
        )
    
    # Plot training and test losses
    plot_loss_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        output_dir=args.output_dir
    )
    
    # Save final model
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    final_path = os.path.join(args.output_dir, "models", "multitask_modo_final.pth")
    torch.save(model_to_save.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Learning with MoDo")
    
    # Data paths
    parser.add_argument("--asr_csv_path", type=str, required=True, help="Path to ASR CSV file")
    parser.add_argument("--st_csv_path", type=str, required=True, help="Path to ST CSV file")
    parser.add_argument("--asr_tokenizer_path", type=str, required=True, help="Path to ASR tokenizer model")
    parser.add_argument("--st_tokenizer_path", type=str, required=True, help="Path to ST tokenizer model")
    parser.add_argument("--cpc_data_dirs", type=str, nargs="+", default=None, help="Directories with audio files for CPC")
    
    # Model parameters
    parser.add_argument("--encoder_dim", type=int, default=512, help="Encoder dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=8, help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--context_frames", type=int, default=240, help="Number of context frames for CPC")
    parser.add_argument("--future_frames", type=int, default=120, help="Number of future frames for CPC")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_interval", type=int, default=5, help="Save model every N epochs")
    
    # MoDo parameters
    parser.add_argument("--modo_gamma", type=float, default=0.1, help="MoDo gamma parameter")
    parser.add_argument("--modo_rho", type=float, default=0.01, help="MoDo rho parameter")
    
    # Other parameters
    parser.add_argument("--use_cpc", action="store_true", help="Whether to use CPC task")
    parser.add_argument("--output_dir", type=str, default="results_modo", help="Output directory")
    parser.add_argument("--analyze_conflicts", action="store_true", help="Whether to analyze task conflicts")
    
    args = parser.parse_args()
    main(args)
