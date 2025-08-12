import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import cosine_similarity
from collections import defaultdict

def set_seed(seed):
    """Set seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return

def projection2simplex(v, s=1.0):
    """Projects vector v onto the simplex with sum s"""
    assert s > 0, "Simplex sum must be positive"
    n = v.shape[0]  # dimension of the vector
    
    # Sort v in descending order
    u, _ = torch.sort(v, descending=True)
    
    # Compute cumulative sum
    cssv = torch.cumsum(u, dim=0)
    
    # Find rho
    rho_candidate = torch.nonzero(u * torch.arange(1, n+1, device=v.device) > (cssv - s))
    
    # Handle empty tensor case
    if rho_candidate.numel() == 0:
        rho = 0  # All values are negative, project to zeros
    else:
        rho = rho_candidate[-1, 0].item()
    
    # Compute theta
    theta = (cssv[rho] - s) / (rho + 1)
    
    # Project
    return torch.clamp(v - theta, min=0.0)

def get_grad_vec(model, param_list=None):
    """
    Get gradient vector from model parameters
    
    Args:
        model: Model with gradients
        param_list: List of parameters to get gradients from (if None, get all)
    """
    grad_vec = []
    
    if param_list is None:
        # Get all parameters
        param_list = model.parameters()
    
    for param in param_list:
        if param.grad is not None:
            grad_vec.append(param.grad.data.view(-1).detach())
    
    if grad_vec:
        return torch.cat(grad_vec)
    else:
        # Return empty tensor if no gradients
        return torch.tensor([], device=next(model.parameters()).device)

def set_grad_vec(model, grad_vec, param_list=None):
    """
    Set gradient vector to model parameters
    
    Args:
        model: Model to set gradients for
        grad_vec: Gradient vector to set
        param_list: List of parameters to set gradients for (if None, set all)
    """
    if param_list is None:
        # Get all parameters
        param_list = list(model.parameters())
    
    pointer = 0
    for param in param_list:
        if param.grad is not None:
            num_param = param.numel()
            param.grad.data = grad_vec[pointer:pointer + num_param].view(param.grad.data.size()).clone()
            pointer += num_param

def grad_modo(grad_list, lambd, gamma=0.1, rho=0.0):
    """
    MoDo (Multi-Objective Descent Oracle) method for computing update direction
    with NaN handling and numerical stability improvements
    
    Args:
        grad_list: List of two tensors with shape [num_tasks, num_params]
        lambd: Current lambda values with shape [num_tasks]
        gamma: Learning rate for lambda update
        rho: Regularization parameter
        
    Returns:
        Updated lambda values and the multi-gradient update direction
    """
    # Check for NaN in lambda and replace with uniform weights if needed
    if torch.isnan(lambd).any():
        print("Warning: NaN detected in input lambda. Resetting to uniform weights.")
        lambd = torch.ones_like(lambd) / lambd.shape[0]
    
    # Project lambda to simplex
    lambd = projection2simplex(lambd)
    
    # Compute multi-gradient from first batch
    multi_grad_1 = lambd @ grad_list[0]
    
    # Check for NaN in multi_grad_1
    if torch.isnan(multi_grad_1).any():
        print("Warning: NaN detected in multi-gradient. Using mean gradient instead.")
        multi_grad_1 = torch.mean(grad_list[0], dim=0)
    
    # Compute gradient of lambda subproblem using second batch
    grad_lambd = torch.zeros_like(lambd)
    
    for i in range(lambd.shape[0]):
        # Calculate dot product with safety checks
        dot_product = torch.sum(multi_grad_1 * grad_list[1][i])
        
        # Check for NaN in dot product
        if torch.isnan(dot_product).item():
            print(f"Warning: NaN detected in dot product for task {i}. Setting to zero.")
            dot_product = torch.tensor(0.0, device=lambd.device)
            
        grad_lambd[i] = dot_product
        
        if rho > 0:
            # Add regularization
            grad_lambd[i] += rho * (lambd[i] - 1.0/lambd.shape[0])
    
    # Check for extreme gradient values and clip if necessary
    if torch.max(torch.abs(grad_lambd)) > 1e6:
        print("Warning: Extremely large gradient detected. Clipping gradients.")
        grad_lambd = torch.clamp(grad_lambd, min=-1e6, max=1e6)
    
    # Update lambda with safety checks
    # First check if grad_lambd has NaN
    if torch.isnan(grad_lambd).any():
        print("Warning: NaN detected in grad_lambd. Setting to zero.")
        grad_lambd = torch.zeros_like(grad_lambd)
        
    # Update lambda
    lambd_new = lambd - gamma * grad_lambd
    
    # Check if new lambda has NaN and reset if needed
    if torch.isnan(lambd_new).any():
        print("Warning: NaN detected in updated lambda. Keeping previous values.")
        lambd_new = lambd.clone()
    
    # Project lambda to simplex
    lambd_new = projection2simplex(lambd_new)
    
    # Compute final multi-gradient using updated lambda
    multi_grad = lambd_new @ grad_list[0]
    
    # Final NaN check for multi_grad
    if torch.isnan(multi_grad).any():
        print("Warning: NaN in final multi-gradient. Using mean gradient.")
        multi_grad = torch.mean(grad_list[0], dim=0)
    
    return lambd_new, multi_grad

def compute_task_loss(model, task_name, inputs, criterion, device):
    """
    Compute loss for a specific task
    
    Args:
        model: Model to compute loss for
        task_name: Name of the task
        inputs: Dictionary of input data
        criterion: Loss function
        device: Device to compute on
        
    Returns:
        Loss value
    """
    if task_name == "ASR":
        # Process ASR task
        features = inputs['features'].to(device)
        feature_lengths = inputs['feat_lens'].to(device)
        asr_targets = inputs[f'ASR_text'].to(device)
        
        out_len, asr_logits = model(x=features, input_lengths=feature_lengths, task="ASR")
        asr_logits = asr_logits.transpose(0, 1)  # (T, N, C)
        target_lengths = torch.sum(asr_targets != 1, dim=1)
        loss = criterion(asr_logits, asr_targets, out_len, target_lengths)
        
    elif task_name == "ST":
        # Process ST task
        features = inputs['features'].to(device)
        feature_lengths = inputs['feat_lens'].to(device)
        trans_targets = inputs[f'ST_text'].to(device)
        
        tgt_input = trans_targets[:, :-1]
        tgt_output = trans_targets[:, 1:]
        st_logits = model(x=features, input_lengths=feature_lengths, tgt_input=tgt_input, task="ST")
        loss = criterion(st_logits.view(-1, st_logits.size(-1)), tgt_output.reshape(-1))
        
    elif task_name == "CPC":
        # Process CPC task
        if 'contexts' in inputs and 'futures' in inputs:
            contexts = inputs['contexts'].to(device)
            futures = inputs['futures'].to(device)
            
            context_lengths = torch.full((contexts.size(0),), contexts.size(1), device=device)
            predicted_future = model(x=contexts, input_lengths=context_lengths, task="CPC")
            
            future_lengths = torch.full((futures.size(0),), futures.size(1), device=device)
            
            # Use the model's encoder to encode the future frames
            # Access encoder directly based on whether model is wrapped in DataParallel
            if isinstance(model, nn.DataParallel):
                future_enc, _ = model.module.encoder(futures, future_lengths)
            else:
                future_enc, _ = model.encoder(futures, future_lengths)
                
            loss = criterion(predicted_future, future_enc)
        else:
            # Handle the CPC data from dedicated dataloader
            contexts, futures, c_lens, f_lens = inputs
            contexts = contexts.to(device)
            futures = futures.to(device)
            c_lens = c_lens.to(device)
            f_lens = f_lens.to(device)
            
            predicted_future = model(x=contexts, input_lengths=c_lens, task="CPC")
            
            if isinstance(model, nn.DataParallel):
                future_enc, _ = model.module.encoder(futures, f_lens)
            else:
                future_enc, _ = model.encoder(futures, f_lens)
                
            loss = criterion(predicted_future, future_enc)
    else:
        raise ValueError(f"Unknown task: {task_name}")
        
    return loss

def compute_task_gradients(model, task_configs, inputs, criterion_dict, device):
    """
    Compute gradients for multiple tasks 
    
    Args:
        model: Model to compute gradients for
        task_configs: Dictionary of task configurations
        inputs: Dictionary of input data
        criterion_dict: Dictionary of loss functions
        device: Device to compute on
        
    Returns:
        task_grads: List of gradient vectors for each task
        task_losses: List of loss values for each task
    """
    task_grads = []
    task_losses = []
    
    # Save original parameters
    original_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_params[name] = param.data.clone()
    
    # Get the encoder parameters (shared parameters)
    shared_params = model.get_shared_parameters()
    
    # Process each task
    for task_name, config in task_configs.items():
        # Reset gradients
        model.zero_grad()
        
        # Get task-specific parameters
        task_params = model.get_task_parameters(task_name)
        
        # Forward pass for this task
        loss = compute_task_loss(model, task_name, inputs, criterion_dict[task_name], device)
        
        # Backward pass
        loss.backward()
        
        # Get gradients for shared parameters only
        shared_grad = get_grad_vec(model, shared_params)
        task_grads.append(shared_grad)
        task_losses.append(loss.item())
        
        # Restore original parameters
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])
    
    # Stack gradients: [num_tasks, num_params]
    return torch.stack(task_grads), torch.tensor(task_losses, device=device)

def update_shared_and_task_specific_params(model, optimizer, task_configs, lambd, multi_grad):
    """
    Update shared parameters using MoDo and task-specific parameters using their gradients
    
    Args:
        model: Model to update
        optimizer: Optimizer for the model
        task_configs: Dictionary of task configurations
        lambd: Lambda weights for MoDo
        multi_grad: Multi-gradient for shared parameters
    """
    # Get shared parameters
    shared_params = model.get_shared_parameters()
    
    # Set multi-gradient for shared parameters
    optimizer.zero_grad()
    set_grad_vec(model, multi_grad, shared_params)
    
    # Update parameters
    optimizer.step()

def compute_pairwise_conflicts(grads):
    """Compute pairwise cosine similarities between task gradients"""
    num_tasks = grads.shape[0]
    conflicts = {}
    
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            grad_i = grads[i]
            grad_j = grads[j]
            
            if torch.norm(grad_i) > 1e-10 and torch.norm(grad_j) > 1e-10:
                cosine_sim = cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0)).item()
            else:
                cosine_sim = 0.0
                
            task_pair = f"Task{i+1}_Task{j+1}"  # ASR_ST, ASR_CPC, ST_CPC
            conflicts[task_pair] = cosine_sim
            
    return conflicts

def analyze_layerwise_conflicts(model, grads, task_names=None):
    """Analyze conflicts by layer for all task pairs"""
    num_tasks = grads.shape[0]
    layer_conflicts = {}
    
    if task_names is None:
        task_names = [f"Task{i+1}" for i in range(num_tasks)]
    
    # Get layer-wise gradients
    layer_grads = {}
    pointer = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            num_param = param.numel()
            
            # Extract gradients for this layer for each task
            layer_grads[name] = []
            for task in range(num_tasks):
                if pointer + num_param <= grads[task].size(0):
                    task_layer_grad = grads[task][pointer:pointer + num_param]
                    layer_grads[name].append(task_layer_grad)
                
            pointer += num_param
    
    # Compute pairwise conflicts for each layer
    for name, layer_task_grads in layer_grads.items():
        if len(layer_task_grads) == num_tasks:  # Ensure we have gradients for all tasks
            layer_conflicts[name] = {}
            
            for i in range(num_tasks):
                for j in range(i+1, num_tasks):
                    grad_i = layer_task_grads[i]
                    grad_j = layer_task_grads[j]
                    
                    if torch.norm(grad_i) > 1e-10 and torch.norm(grad_j) > 1e-10:
                        cosine_sim = cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0)).item()
                    else:
                        cosine_sim = 0.0
                        
                    task_pair = f"{task_names[i]}_{task_names[j]}"
                    layer_conflicts[name][task_pair] = cosine_sim
    
    return layer_conflicts

def prettify_layer_name(name):
    """Create nice layer names for plots"""
    if 'encoder.conv_modules' in name:
        layer_num = name.split('encoder.conv_modules.')[1].split('.')[0]
        if 'conv' in name and 'weight' in name:
            return f'Encoder_Conv{layer_num}'
    elif 'encoder.attention_module' in name:
        if 'weight' in name:
            return 'Encoder_Attention'
    elif 'encoder.linear' in name:
        if 'weight' in name:
            return 'Encoder_Linear'
    elif '.head.' in name:
        task_name = name.split('.')[0]
        layer_num = name.split('.head.')[1].split('.')[0]
        if layer_num.isdigit():
            if int(layer_num) % 2 == 1:  # Linear layers are odd-indexed
                return f'{task_name}_Linear{(int(layer_num)//2)+1}'
    elif 'decoder' in name:
        if 'encoder_layers' in name:
            layer_num = name.split('encoder_layers.')[1].split('.')[0]
            return f'Decoder_Enc{layer_num}'
        elif 'decoder_layers' in name:
            layer_num = name.split('decoder_layers.')[1].split('.')[0]
            return f'Decoder_Dec{layer_num}'
    elif 'output_projection' in name:
        task_name = name.split('.')[0]
        return f'{task_name}_Output'
    elif 'embedding' in name:
        task_name = name.split('.')[0]
        return f'{task_name}_Embedding'
    elif 'predictor' in name:
        task_name = name.split('.')[0]
        layer_num = name.split('predictor.')[1].split('.')[0]
        if layer_num.isdigit():
            if int(layer_num) % 2 == 1:  # Linear layers are odd-indexed
                return f'{task_name}_Linear{(int(layer_num)//2)+1}'
    
    if '.bias' in name:
        base = name.split('.bias')[0]
        return f"{base}_bias"
        
    return name.replace('.', '_')

# --------------------------
# VISUALIZATION FUNCTIONS
# --------------------------
def plot_conflict_heatmap(all_conflicts, epoch_num, output_dir="conflict_results", task_names=None):
    """Plot heatmap of conflicts between tasks"""
    if task_names is None:
        task_names = [f"Task{i+1}" for i in range(len(all_conflicts) + 1)]
    
    # Create directory
    os.makedirs(f"{output_dir}/epoch_{epoch_num}", exist_ok=True)
    
    # Convert conflicts dictionary to matrix
    num_tasks = len(task_names)
    conflict_matrix = np.ones((num_tasks, num_tasks))
    
    # Fill in the conflicts (upper triangle)
    idx = 0
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            task_pair = f"Task{i+1}_Task{j+1}"
            if task_pair in all_conflicts:
                conflict_matrix[i, j] = all_conflicts[task_pair]
                conflict_matrix[j, i] = all_conflicts[task_pair]  # Mirror
            idx += 1
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create colormap that distinguishes conflict/alignment
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(conflict_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                xticklabels=task_names, yticklabels=task_names)
    
    plt.title(f'Task Conflicts (Cosine Similarity) - Epoch {epoch_num}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch_num}/task_conflicts.png", dpi=300)
    plt.close()

def plot_top_conflicts_by_task_pair(layer_conflicts, epoch_num, output_dir="conflict_results", task_names=None):
    """Plot top 10 conflicting layers for each task pair"""
    if task_names is None:
        # Extract unique task names from layer conflicts
        all_pairs = set()
        for layer_name, conflicts in layer_conflicts.items():
            all_pairs.update(conflicts.keys())
        
        task_names = set()
        for pair in all_pairs:
            for task in pair.split('_'):
                task_names.add(task)
        
        task_names = sorted(list(task_names))
    
    # Create directory
    os.makedirs(f"{output_dir}/epoch_{epoch_num}", exist_ok=True)
    
    # Get task pairs from all layer conflicts
    task_pairs = set()
    for layer_name, conflicts in layer_conflicts.items():
        task_pairs.update(conflicts.keys())
    
    task_pairs = sorted(list(task_pairs))
    
    # For each task pair, find top conflicting layers
    for task_pair in task_pairs:
        # Get conflicts for this task pair
        pair_conflicts = {}
        for layer_name, conflicts in layer_conflicts.items():
            if task_pair in conflicts:
                pair_conflicts[layer_name] = conflicts[task_pair]
        
        # Skip if no conflicts found
        if not pair_conflicts:
            continue
            
        # Sort by absolute value of conflict
        sorted_conflicts = sorted(pair_conflicts.items(), key=lambda x: abs(x[1]), reverse=True)
        top_conflicts = sorted_conflicts[:10]
        
        # Pretty names and values
        layer_names = [prettify_layer_name(name) for name, _ in top_conflicts]
        conflict_values = [val for _, val in top_conflicts]
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        ## Colors based on conflict strength
        colors = []
        for val in conflict_values:
            if val < -0.5:
                colors.append('#d62728')  # Strong conflict (dark red)
            elif val < 0:
                colors.append('#ff9896')  # Mild conflict (light red)
            elif val < 0.5:
                colors.append('#aec7e8')  # Mild alignment (light blue)
            else:
                colors.append('#1f77b4')  # Strong alignment (dark blue)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(layer_names))
        bars = plt.barh(y_pos, conflict_values, color=colors)
        
        # Add labels and title
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        plt.axvspan(0, 1.1, alpha=0.1, color='blue', label='Alignment Zone')
        plt.axvspan(-1.1, 0, alpha=0.1, color='red', label='Conflict Zone')
        plt.yticks(y_pos, layer_names)
        plt.xlabel('Cosine Similarity (Conflict < 0, Alignment > 0)')
        plt.title(f'Top 10 Most Conflicting Layers: {task_pair} - Epoch {epoch_num}')
        plt.xlim(-1.1, 1.1)
        
        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ha = 'left' if width < 0 else 'right'
            x_pos = width - 0.1 if width < 0 else width + 0.1
            plt.text(x_pos, bar.get_y() + bar.get_height()/2, 
                    f"{width:.2f}", va='center', ha=ha, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', label='Strong Conflict (cos < -0.5)'),
            Patch(facecolor='#ff9896', label='Mild Conflict (cos < 0)'),
            Patch(facecolor='#aec7e8', label='Mild Alignment (0 < cos < 0.5)'),
            Patch(facecolor='#1f77b4', label='Strong Alignment (cos ≥ 0.5)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/epoch_{epoch_num}/{task_pair}_conflicts.png", dpi=300)
        plt.close()

def plot_lambda_evolution(lambda_history, output_dir="conflict_results", task_names=None):
    """Plot evolution of lambda weights throughout training"""
    if task_names is None:
        task_names = [f"Task{i+1}" for i in range(lambda_history.shape[1])]
    
    # Create directory
    os.makedirs(f"{output_dir}/final", exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Convert lambda history to numpy array if needed
    if isinstance(lambda_history, list):
        lambda_history = np.array(lambda_history)
    
    # Plot lambda values
    epochs = range(1, len(lambda_history) + 1)
    for i, task_name in enumerate(task_names):
        plt.plot(epochs, lambda_history[:, i], '-', label=f'{task_name} Weight (λ{i+1})')
    
    # Add labels and title
    plt.title('MoDo Task Weights During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final/lambda_weights.png", dpi=300)
    plt.close()

def plot_loss_curves(train_losses, test_losses, output_dir="conflict_results"):
    """
    Plot training and test losses
    
    Args:
        train_losses: Dictionary of training losses for each task
        test_losses: Dictionary of test losses for each task
        output_dir: Directory to save the plot
    """
    # Create directory
    os.makedirs(f"{output_dir}/final", exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Get number of epochs
    num_epochs = max(len(losses) for losses in train_losses.values())
    epochs = range(1, num_epochs + 1)
    
    # Plot each task's losses
    for task_name, losses in train_losses.items():
        if task_name == "ASR":
            color = 'b'
        elif task_name == "ST":
            color = 'r'
        elif task_name == "CPC":
            color = 'g'
        else:
            color = None  # Let matplotlib choose color
            
        plt.plot(epochs, losses, f'{color}-', label=f'Train {task_name} Loss')
        
    for task_name, losses in test_losses.items():
        if task_name == "ASR":
            color = 'b'
        elif task_name == "ST":
            color = 'r'
        elif task_name == "CPC":
            color = 'g'
        else:
            color = None  # Let matplotlib choose color
            
        plt.plot(epochs, losses, f'{color}--', label=f'Test {task_name} Loss')
    
    # Add labels and title
    plt.title('Task Losses During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final/loss_curves.png", dpi=300)
    plt.close()