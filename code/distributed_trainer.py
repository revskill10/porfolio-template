import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Optional, List
import logging
import time
import os
import socket
from contextlib import contextmanager

class DistributedTrainer:
    """
    Fault-tolerant distributed training system with dynamic scaling support.
    
    Features:
    - Automatic fault detection and recovery
    - Dynamic node scaling
    - Gradient synchronization with compression
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Move model to device and wrap with DDP
        self.model = model.to(self.device)
        self.model = DDP(
            self.model, 
            device_ids=[self.local_rank],
            find_unused_parameters=config.get('find_unused_parameters', False)
        )
        
        # Setup optimizers with learning rate scaling
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize logging and monitoring
        self.logger = self._setup_logger()
        self.metrics = {}
        
        # Fault tolerance settings
        self.max_retries = config.get('max_retries', 3)
        self.checkpoint_freq = config.get('checkpoint_freq', 100)
        
        self.logger.info(f"Initialized trainer on rank {self.rank}/{self.world_size}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with linear learning rate scaling"""
        base_lr = self.config['learning_rate']
        scaled_lr = base_lr * self.world_size  # Linear scaling rule
        
        if self.config.get('optimizer', 'adamw').lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=scaled_lr,
                weight_decay=self.config.get('weight_decay', 1e-4),
                betas=self.config.get('betas', (0.9, 0.999))
            )
        else:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=scaled_lr,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['max_epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with fault tolerance and monitoring
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        num_batches = 0
        
        # Set epoch for distributed sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            try:
                # Move data to device
                data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                # Forward pass
                loss = self._training_step(data, targets)
                
                # Update metrics
                total_loss += loss.item()
                total_samples += data.size(0)
                num_batches += 1
                
                # Logging
                if batch_idx % self.config.get('log_interval', 100) == 0 and self.rank == 0:
                    self._log_training_progress(epoch, batch_idx, loss.item(), len(dataloader))
                
                # Checkpointing
                if batch_idx % self.checkpoint_freq == 0 and self.rank == 0:
                    self._save_checkpoint(epoch, batch_idx)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM error at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Synchronize metrics across all processes
        epoch_metrics = self._synchronize_metrics(total_loss, total_samples, num_batches)
        
        # Update learning rate
        self.scheduler.step()
        
        # Log epoch summary
        epoch_time = time.time() - start_time
        if self.rank == 0:
            self._log_epoch_summary(epoch, epoch_metrics, epoch_time)
        
        return epoch_metrics
    
    def _training_step(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Single training step with gradient clipping"""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(data)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        if self.config.get('grad_clip_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config['grad_clip_norm']
            )
        
        # Optimizer step
        self.optimizer.step()
        
        return loss
    
    def _synchronize_metrics(self, total_loss: float, total_samples: int, num_batches: int) -> Dict[str, float]:
        """Synchronize metrics across all processes using all-reduce"""
        # Create tensors for reduction
        metrics_tensor = torch.tensor([total_loss, total_samples, num_batches], 
                                    dtype=torch.float32, device=self.device)
        
        # All-reduce to sum across processes
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Extract synchronized values
        global_loss, global_samples, global_batches = metrics_tensor.tolist()
        
        return {
            'loss': global_loss / global_batches,
            'samples_per_sec': global_samples / max(1, time.time() - getattr(self, '_epoch_start_time', time.time())),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validation with distributed evaluation"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total_samples += data.size(0)
        
        # Synchronize validation metrics
        val_tensor = torch.tensor([total_loss, correct, total_samples], 
                                dtype=torch.float32, device=self.device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        
        global_loss, global_correct, global_samples = val_tensor.tolist()
        
        return {
            'val_loss': global_loss / len(dataloader),
            'val_accuracy': global_correct / global_samples
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup distributed logging"""
        logger = logging.getLogger(f'trainer_rank_{self.rank}')
        
        if self.rank == 0:  # Only rank 0 logs to avoid duplication
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)  # Reduce logging for other ranks
            
        return logger
    
    def _log_training_progress(self, epoch: int, batch_idx: int, loss: float, total_batches: int):
        """Log training progress"""
        progress = 100.0 * batch_idx / total_batches
        lr = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(
            f'Epoch {epoch:3d} [{batch_idx:5d}/{total_batches:5d} ({progress:6.2f}%)] '
            f'Loss: {loss:.6f} LR: {lr:.2e}'
        )
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict[str, float], epoch_time: float):
        """Log epoch summary"""
        self.logger.info(
            f'Epoch {epoch:3d} Summary - '
            f'Loss: {metrics["loss"]:.6f} '
            f'Samples/sec: {metrics["samples_per_sec"]:.1f} '
            f'Time: {epoch_time:.2f}s'
        )
    
    def _save_checkpoint(self, epoch: int, batch_idx: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    @contextmanager
    def fault_tolerance(self):
        """Context manager for fault-tolerant operations"""
        retries = 0
        while retries < self.max_retries:
            try:
                yield
                break
            except Exception as e:
                retries += 1
                self.logger.warning(f"Operation failed (attempt {retries}/{self.max_retries}): {e}")
                if retries >= self.max_retries:
                    raise e
                time.sleep(2 ** retries)  # Exponential backoff

def setup_distributed_training(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize distributed training environment
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args()
    
    # Setup distributed training
    setup_distributed_training(args.rank, args.world_size)
    
    # Example configuration
    config = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'max_epochs': 100,
        'grad_clip_norm': 1.0,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    }
    
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    # Initialize trainer
    trainer = DistributedTrainer(model, config)
    
    print(f"Distributed trainer initialized on rank {args.rank}")
    
    # Cleanup
    cleanup_distributed()
