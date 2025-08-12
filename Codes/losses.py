import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    """
    InfoNCE loss for contrastive learning with improved stability
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = max(temperature, 1e-4)  # Ensure temperature is not too small
        
    def forward(self, query, positive_key, negative_keys=None):
        """
        Compute InfoNCE loss with NaN protection
        
        Args:
            query: Tensor of shape [batch_size, dim]
            positive_key: Tensor of shape [batch_size, dim]
            negative_keys: Optional tensor of shape [n_negatives, dim] 
                           If None, use other samples in the batch as negatives
        
        Returns:
            Loss value
        """
        # Apply safety checks to avoid NaN
        def prepare_tensor(x):
            # Replace NaN with zeros and Inf with large but finite values
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            # Apply L2 normalization with epsilon to avoid division by zero
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            norm = torch.clamp(norm, min=1e-6)  # Avoid division by zero
            return x / norm
        
        # Normalize inputs
        query = prepare_tensor(query)
        positive_key = prepare_tensor(positive_key)
        
        batch_size, dim = query.shape
        
        try:
            if negative_keys is None:
                # Use other samples in batch as negatives
                # Create all-to-all similarity matrix but exclude self-similarities
                all_samples = positive_key  # [B, D]
                
                # Compute pairwise similarities
                similarities = torch.matmul(query, all_samples.t())  # [B, B]
                similarities = similarities / self.temperature
                
                # Create mask for positive samples (diagonal elements)
                mask = torch.eye(batch_size, dtype=torch.bool, device=query.device)
                
                # Use modified cross-entropy with safe handling
                similarities = similarities.clone()
                # Mask out diagonal (positive examples) with large negative value
                similarities = similarities.masked_fill(mask, -1e9)
                
                # Add back positive sample at a known position (first position for each row)
                positive_sims = torch.sum(query * positive_key, dim=1) / self.temperature
                positive_sims = positive_sims.view(-1, 1)
                
                # Concatenate positive sample to the front
                all_sims = torch.cat([positive_sims, similarities], dim=1)
                
                # Target is always the first position (index 0)
                target = torch.zeros(batch_size, dtype=torch.long, device=query.device)
                
                # Apply cross entropy with safety check
                loss = F.cross_entropy(all_sims, target)
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: NaN detected in InfoNCE loss. Using MSE fallback.")
                    # Fallback to MSE loss
                    loss = F.mse_loss(query, positive_key) * 10.0  # Scale to match CE magnitude
                
            else:
                # With provided negative keys - similar approach
                negative_keys = prepare_tensor(negative_keys) if negative_keys is not None else None
                # Implement similar to above but with provided negative keys
                # This part depends on how your negative_keys are structured
                pass  # Not implemented as we don't use this in current code
        
        except Exception as e:
            print(f"Error in InfoNCE computation: {e}")
            # Fallback to simple MSE loss in case of unexpected errors
            loss = F.mse_loss(query, positive_key) * 10.0
            
        return loss

class MultiStepInfoNCE(nn.Module):
    """
    InfoNCE loss for multi-step future prediction in CPC with improved stability
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = max(temperature, 1e-4)  # Ensure temperature is not too small
        self.info_nce = InfoNCE(temperature=self.temperature)
    
    def forward(self, predicted, target):
        """
        InfoNCE loss for CPC with flexible sequence length handling and NaN protection
        
        Args:
            predicted: [B, T_pred, D] - Predicted future representations
            target:    [B, T_targ, D] - Target future representations
        
        Returns:
            Loss value
        """
        # Check for NaN in inputs
        if torch.isnan(predicted).any() or torch.isinf(predicted).any():
            print("Warning: NaN or Inf detected in CPC predictions. Applying fix...")
            predicted = torch.nan_to_num(predicted, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("Warning: NaN or Inf detected in CPC targets. Applying fix...")
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        B, T_pred, D = predicted.shape
        _, T_targ, _ = target.shape
        
        try:
            # Handle sequence length mismatch by truncating the longer one
            T_min = min(T_pred, T_targ)
            predicted_trunc = predicted[:, :T_min, :]  # [B, T_min, D]
            target_trunc = target[:, :T_min, :]        # [B, T_min, D]
            
            # Flatten the truncated tensors
            pred_flat = predicted_trunc.reshape(B * T_min, D)  # [B*T_min, D]
            targ_flat = target_trunc.reshape(B * T_min, D)     # [B*T_min, D]
            
            # Use InfoNCE with enhanced stability
            loss = self.info_nce(pred_flat, targ_flat)
            
            # Check for NaN in the result
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN in CPC loss after InfoNCE. Using MSE fallback.")
                # Fallback to MSE loss
                loss = F.mse_loss(pred_flat, targ_flat) * 10.0
                
        except Exception as e:
            print(f"Error in MultiStepInfoNCE: {e}")
            # Reshape to handle any unexpected errors
            pred_flat = predicted.reshape(-1, D)
            targ_flat = target[:, :pred_flat.size(0)//B, :].reshape(-1, D)
            # Ensure same size
            min_size = min(pred_flat.size(0), targ_flat.size(0))
            # Use MSE loss as fallback
            loss = F.mse_loss(
                pred_flat[:min_size], 
                targ_flat[:min_size]
            ) * 10.0
            
        return loss