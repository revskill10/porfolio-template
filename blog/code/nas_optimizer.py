import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict

class DifferentiableNAS(nn.Module):
    """
    Differentiable Neural Architecture Search implementation
    
    This implementation follows the DARTS approach with continuous relaxation
    of the architecture search space, enabling gradient-based optimization.
    """
    
    def __init__(self, search_space: List[str], num_layers: int = 8, num_nodes: int = 4):
        super().__init__()
        self.search_space = search_space
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        
        # Architecture parameters (learnable)
        self.alpha = nn.Parameter(
            torch.randn(num_layers, num_nodes, len(search_space))
        )
        
        # Operation candidates
        self.ops = nn.ModuleList([
            self._get_operation(op_name) 
            for op_name in search_space
        ])
        
        # Initialize architecture parameters
        self._initialize_alpha()
    
    def _initialize_alpha(self):
        """Initialize architecture parameters with small random values"""
        with torch.no_grad():
            self.alpha.normal_(0, 1e-3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with architecture sampling"""
        for layer_idx in range(self.num_layers):
            x = self._forward_layer(x, layer_idx)
        return x
    
    def _forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Forward pass through a single layer with mixed operations"""
        layer_outputs = []
        
        for node_idx in range(self.num_nodes):
            # Softmax over architecture choices for this node
            weights = F.softmax(self.alpha[layer_idx, node_idx], dim=0)
            
            # Weighted combination of operations
            node_output = sum(
                w * op(x) for w, op in zip(weights, self.ops)
            )
            layer_outputs.append(node_output)
        
        # Combine node outputs (simple concatenation + projection)
        if len(layer_outputs) > 1:
            combined = torch.cat(layer_outputs, dim=1)
            # Project back to original dimension
            return F.linear(combined, 
                          torch.randn(x.size(-1), combined.size(-1), device=x.device))
        else:
            return layer_outputs[0]
    
    def get_best_architecture(self) -> List[List[int]]:
        """Extract the best architecture after training"""
        with torch.no_grad():
            best_arch = []
            for layer_idx in range(self.num_layers):
                layer_arch = []
                for node_idx in range(self.num_nodes):
                    best_op = torch.argmax(self.alpha[layer_idx, node_idx]).item()
                    layer_arch.append(best_op)
                best_arch.append(layer_arch)
            return best_arch
    
    def get_architecture_weights(self) -> torch.Tensor:
        """Get current architecture weights (softmax of alpha)"""
        return F.softmax(self.alpha, dim=-1)
    
    def _get_operation(self, op_name: str) -> nn.Module:
        """Factory method for creating operations"""
        ops = {
            'conv3x3': nn.Conv2d(64, 64, 3, padding=1),
            'conv5x5': nn.Conv2d(64, 64, 5, padding=2),
            'conv1x1': nn.Conv2d(64, 64, 1),
            'maxpool3x3': nn.MaxPool2d(3, stride=1, padding=1),
            'avgpool3x3': nn.AvgPool2d(3, stride=1, padding=1),
            'skip': nn.Identity(),
            'zero': ZeroOperation(),
            'sep_conv3x3': SeparableConv2d(64, 64, 3, padding=1),
            'sep_conv5x5': SeparableConv2d(64, 64, 5, padding=2),
            'dil_conv3x3': nn.Conv2d(64, 64, 3, padding=2, dilation=2),
        }
        return ops.get(op_name, nn.Identity())

class ZeroOperation(nn.Module):
    """Zero operation that returns zeros"""
    def forward(self, x):
        return torch.zeros_like(x)

class SeparableConv2d(nn.Module):
    """Separable convolution (depthwise + pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class NASOptimizer:
    """
    Bilevel optimizer for Neural Architecture Search
    
    Implements the DARTS optimization strategy with separate optimizers
    for architecture parameters and network weights.
    """
    
    def __init__(self, model: DifferentiableNAS, 
                 w_lr: float = 0.025, alpha_lr: float = 3e-4,
                 w_momentum: float = 0.9, w_weight_decay: float = 3e-4):
        self.model = model
        
        # Separate parameters for weights and architecture
        model_params = []
        alpha_params = []
        
        for name, param in model.named_parameters():
            if 'alpha' in name:
                alpha_params.append(param)
            else:
                model_params.append(param)
        
        # Separate optimizers for weights and architecture
        self.w_optimizer = torch.optim.SGD(
            model_params, lr=w_lr, momentum=w_momentum, weight_decay=w_weight_decay
        )
        self.alpha_optimizer = torch.optim.Adam(
            alpha_params, lr=alpha_lr, betas=(0.5, 0.999), weight_decay=1e-3
        )
        
        # Learning rate schedulers
        self.w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_optimizer, T_max=50, eta_min=1e-4
        )
        self.alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.alpha_optimizer, T_max=50, eta_min=1e-5
        )
    
    def step(self, train_data: torch.Tensor, train_targets: torch.Tensor,
             val_data: torch.Tensor, val_targets: torch.Tensor) -> Tuple[float, float]:
        """
        Perform one step of bilevel optimization
        
        Returns:
            Tuple of (train_loss, val_loss)
        """
        # Step 1: Update architecture parameters on validation data
        alpha_loss = self._update_alpha(val_data, val_targets)
        
        # Step 2: Update network weights on training data  
        w_loss = self._update_weights(train_data, train_targets)
        
        return w_loss, alpha_loss
    
    def _update_alpha(self, val_data: torch.Tensor, val_targets: torch.Tensor) -> float:
        """Update architecture parameters using validation data"""
        self.alpha_optimizer.zero_grad()
        self.model.train()
        
        # Forward pass on validation data
        outputs = self.model(val_data)
        loss = F.cross_entropy(outputs, val_targets)
        
        # Backward pass and update
        loss.backward()
        self.alpha_optimizer.step()
        
        return loss.item()
    
    def _update_weights(self, train_data: torch.Tensor, train_targets: torch.Tensor) -> float:
        """Update network weights using training data"""
        self.w_optimizer.zero_grad()
        self.model.train()
        
        # Forward pass on training data
        outputs = self.model(train_data)
        loss = F.cross_entropy(outputs, train_targets)
        
        # Backward pass and update
        loss.backward()
        self.w_optimizer.step()
        
        return loss.item()
    
    def step_schedulers(self):
        """Step both learning rate schedulers"""
        self.w_scheduler.step()
        self.alpha_scheduler.step()
    
    def get_current_lr(self) -> Dict[str, float]:
        """Get current learning rates"""
        return {
            'weight_lr': self.w_optimizer.param_groups[0]['lr'],
            'alpha_lr': self.alpha_optimizer.param_groups[0]['lr']
        }

class ArchitectureEvaluator:
    """
    Evaluates discovered architectures and provides analysis
    """
    
    def __init__(self, search_space: List[str]):
        self.search_space = search_space
        self.evaluation_history = []
    
    def evaluate_architecture(self, architecture: List[List[int]], 
                            test_loader, device: str = 'cuda') -> Dict[str, float]:
        """
        Evaluate a specific architecture on test data
        
        Args:
            architecture: List of operation indices for each layer/node
            test_loader: DataLoader for test data
            device: Device to run evaluation on
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Build model from architecture
        model = self._build_model_from_architecture(architecture)
        model = model.to(device)
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                loss = F.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total_samples += data.size(0)
        
        metrics = {
            'accuracy': correct / total_samples,
            'loss': total_loss / len(test_loader),
            'parameters': sum(p.numel() for p in model.parameters()),
            'flops': self._estimate_flops(model, next(iter(test_loader))[0][:1])
        }
        
        self.evaluation_history.append({
            'architecture': architecture,
            'metrics': metrics
        })
        
        return metrics
    
    def _build_model_from_architecture(self, architecture: List[List[int]]) -> nn.Module:
        """Build a concrete model from architecture specification"""
        # This is a simplified version - in practice, you'd build the full model
        layers = []
        for layer_arch in architecture:
            # Use the most frequent operation in this layer
            op_counts = defaultdict(int)
            for op_idx in layer_arch:
                op_counts[op_idx] += 1
            
            most_common_op = max(op_counts.items(), key=lambda x: x[1])[0]
            op_name = self.search_space[most_common_op]
            
            # Add the operation (simplified)
            if 'conv' in op_name:
                layers.append(nn.Conv2d(64, 64, 3, padding=1))
                layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _estimate_flops(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Estimate FLOPs for the model (simplified)"""
        # This is a very simplified FLOP estimation
        total_params = sum(p.numel() for p in model.parameters())
        input_size = sample_input.numel()
        # Rough approximation: 2 * params * input_size
        return 2 * total_params * input_size
    
    def get_pareto_frontier(self) -> List[Dict]:
        """Get Pareto-optimal architectures (accuracy vs efficiency)"""
        if not self.evaluation_history:
            return []
        
        pareto_architectures = []
        
        for i, eval_result in enumerate(self.evaluation_history):
            is_pareto = True
            metrics_i = eval_result['metrics']
            
            for j, other_result in enumerate(self.evaluation_history):
                if i == j:
                    continue
                
                metrics_j = other_result['metrics']
                
                # Check if j dominates i (higher accuracy, fewer parameters)
                if (metrics_j['accuracy'] >= metrics_i['accuracy'] and 
                    metrics_j['parameters'] <= metrics_i['parameters'] and
                    (metrics_j['accuracy'] > metrics_i['accuracy'] or 
                     metrics_j['parameters'] < metrics_i['parameters'])):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_architectures.append(eval_result)
        
        return pareto_architectures

# Example usage
if __name__ == "__main__":
    # Define search space
    search_space = [
        'conv3x3', 'conv5x5', 'sep_conv3x3', 'sep_conv5x5',
        'maxpool3x3', 'avgpool3x3', 'skip', 'zero'
    ]
    
    # Create NAS model
    nas_model = DifferentiableNAS(search_space, num_layers=6, num_nodes=4)
    
    # Create optimizer
    optimizer = NASOptimizer(nas_model)
    
    # Print model info
    total_params = sum(p.numel() for p in nas_model.parameters())
    alpha_params = nas_model.alpha.numel()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture parameters: {alpha_params:,}")
    print(f"Search space size: {len(search_space)}")
    print(f"Architecture space size: {len(search_space)**(nas_model.num_layers * nas_model.num_nodes)}")
    
    # Example training step (with dummy data)
    dummy_train_data = torch.randn(32, 64, 32, 32)
    dummy_train_targets = torch.randint(0, 10, (32,))
    dummy_val_data = torch.randn(16, 64, 32, 32)
    dummy_val_targets = torch.randint(0, 10, (16,))
    
    train_loss, val_loss = optimizer.step(
        dummy_train_data, dummy_train_targets,
        dummy_val_data, dummy_val_targets
    )
    
    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    
    # Get best architecture
    best_arch = nas_model.get_best_architecture()
    print(f"Best architecture: {best_arch}")
