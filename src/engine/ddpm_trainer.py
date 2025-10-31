import os
from typing import List, Tuple, Dict, Callable, Optional, Union

import numpy as np
from tqdm.auto import tqdm
from rich.progress import (Progress, TextColumn, BarColumn, 
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.console import Console
from PIL import Image
import torchvision.transforms as T

import torch
from torch import nn
from torch.distributed import all_gather, all_gather_object
from torch.utils.data.distributed import DistributedSampler

from .vanilla_trainer import Trainer
from ..utils import cleanup_ddp


class DiffusionTrainer(Trainer):
    """
    Trainer specifically for autoencoder models.
    """
    def __init__(self, 
                 noise_scheduler: Optional[Dict] = None,
                 *args, **kwargs
    ):
        super(DiffusionTrainer, self).__init__(*args, **kwargs)

        # Get noise_scheduler and its attributes
        if noise_scheduler is None: 
            raise ValueError("Noise scheduler must be provided for DDPM")
        self.noise_scheduler = noise_scheduler

        self.num_steps = getattr(self.noise_scheduler, 'num_time_steps', None)
        if self.num_steps is None: 
            raise AttributeError("Noise scheduler does not have attribute 'num_time_steps'")
        
        self.alpha = getattr(self.noise_scheduler, 'alpha', None)
        if self.alpha is None: 
            raise AttributeError("Noise scheduler does not have attribute 'alpha'")
        
        self.beta = getattr(self.noise_scheduler, 'beta', None)
        if self.beta is None:
            raise AttributeError("Noise scheduler does not have attribute 'beta'")

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        try:
            # Callback before training
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self.start_epoch * len(self.train_loader)

            if self.progress_bar and self.rank == 0:
                console = Console()
                progress = Progress(
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    BarColumn(),
                    TextColumn("Epoch {task.fields[epoch]}/{task.fields[total_epoch]}"),
                    TextColumn("Step {task.fields[step]}/{task.fields[total_steps]}"),
                    TextColumn("Loss: {task.fields[avg_loss]:.4f}"),
                    TextColumn("Metric: {task.fields[avg_metric]:.4f}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=True,
                    refresh_per_second=5,
                )
                progress.start()
                task = progress.add_task(
                    "Training", 
                    total=total_steps, 
                    completed=start_step,
                    epoch=self.start_epoch + 1,
                    total_epoch=self.num_epochs,
                    step=0, 
                    total_steps=total_steps,
                    avg_loss=0.0, 
                    avg_metric=0.0
                )

            else:
                class _NoOpBar:
                    def advance(self, *args, **kwargs): pass
                    def update(self, *args, **kwargs): pass
                    def stop(self): pass
                    def update_task(self, *args, **kwargs): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                progress = _NoOpBar()
                task = None

            for epoch in range(self.start_epoch, self.num_epochs):
                # Make DistributedSampler shuffle with a different seed each epoch
                if self._is_distributed and isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)

                # Callback at the beginning of each epoch
                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch, trainer=self)

                # Training phase
                self.model.train()
                running_loss_sum = 0.0
                running_metric_sum = 0.0
                running_count = 0

                for batch_idx, (X,_) in enumerate(self.train_loader):
                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)
                    t = torch.randint(0,self.num_steps,(X.shape[0],))
                    e = torch.randn_like(X, requires_grad=False)
                    a = self.alpha[t].view(X.shape[0], 1, 1, 1).to(self.device)
                    X = (torch.sqrt(a)*X) + (torch.sqrt(1-a)*e)

                    self.optimizer.zero_grad()
                    outputs = self.model(X, t)
                    loss = self.criterion(outputs, e)
                    loss.backward()
                    self.optimizer.step()
                    bsz = X.size(0)
                    running_loss_sum += float(loss.item()) * bsz

                    if self.metric:
                        running_metric_sum += float(self.metric(outputs, X)) * bsz

                    running_count += bsz

                    avg_loss = running_loss_sum / max(running_count, 1)
                    avg_metric = running_metric_sum / max(running_count, 1)

                    # Short summary

                    if self.rank == 0 and task is not None:
                        progress.update(
                        task,
                        advance=1,
                        epoch=epoch,
                        step=step,
                        avg_loss=avg_loss,
                        avg_metric=avg_metric,
                        )

                        if step % self.logging_steps == 0 or step == total_steps:
                            message = (
                                f"step: {step}/{total_steps} | "
                                f"train_loss: {avg_loss:.4f} | "
                                f"train_metric: {avg_metric:.4f}"
                            )
                            console.log(message)

                # Validation phase
                self.model.eval()
                val_loss_sum = 0.0
                val_metric_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for X_val,_ in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                        t_val = torch.randint(0,self.num_steps,(X_val.shape[0],))
                        e_val = torch.randn_like(X_val, requires_grad=False)
                        a_val = self.alpha[t_val].view(X_val.shape[0], 1, 1, 1).to(self.device)

                        X_val = (torch.sqrt(a_val)*X_val) + (torch.sqrt(1-a_val)*e_val)

                        outputs_val = self.model(X_val, t_val)
                        loss_val = self.criterion(outputs_val, X_val)
                        bsz = X_val.size(0)
                        val_loss_sum += float(loss_val.item()) * bsz

                        if self.metric:
                            val_metric_sum += float(self.metric(outputs_val, X_val)) * bsz

                        val_count += bsz
                
                # Gather validation results from all processes
                if self._is_distributed:
                    packed = torch.tensor(
                        data=[val_loss_sum, val_metric_sum, float(val_count)],
                        dtype=torch.float64,
                        device=self.device,
                    )
                    gathered = [torch.zeros_like(packed) for _ in range(self.world_size)]
                    all_gather(gathered, packed)

                    total_loss_sum = sum(g[0].item() for g in gathered)
                    total_metric_sum = sum(g[1].item() for g in gathered)
                    total_count = int(sum(g[2].item() for g in gathered))
                else:
                    total_loss_sum = val_loss_sum
                    total_metric_sum = val_metric_sum
                    total_count = val_count

                # Global averages
                val_loss = total_loss_sum / max(total_count, 1)
                val_metric = (total_metric_sum / max(total_count, 1)) if self.metric else 0.0

                # Short summary for validation
                if self.rank == 0:
                    tqdm.write(
                        f"epoch: {epoch + 1}/{self.num_epochs} | "
                        f"val_loss: {val_loss:.4f} | "
                        f"val_metric: {val_metric:.4f}"
                    )

                if self.scheduler:
                    self.scheduler.step()

                # Get learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best model
                if self.best_model_path and self.rank == 0 and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    to_save = self.model.module if self._is_distributed else self.model
                    torch.save(to_save.state_dict(), self.best_model_path)

                # Update history
                self.history['epoch'].append(epoch + 1)
                self.history['train_loss'].append(avg_loss)
                self.history['train_metric'].append(avg_metric)
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)

                # Save a checkpoint every epoch
                self.save_checkpoint(epoch)

                # Log results
                logs = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'train_metric': avg_metric,
                    'val_loss': val_loss,
                    'val_metric': val_metric,
                    'learning_rate': current_lr,
                }
                self.log_csv(logs)

                # Callback after each epoch
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, trainer=self, logs=logs)

                # Break if any callback says to stop
                if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                    break

            # Callback after training
            for cb in self.callbacks:
                cb.on_train_end(trainer=self)
        except KeyboardInterrupt:
            if self.rank == 0:
                print(f"\nTraining interrupted at epoch {epoch + 1}.")
                
            cleanup_ddp()

        return self.history, self.model
    
    @torch.no_grad()
    def inference(
        self, 
        num_images: Optional[int] = 10,
        image_size: Optional[int] = 64,
        saving_steps: Optional[List[int]] = [
            0, 15, 50, 100, 200, 300, 400, 550, 700, 999
        ]
    ):
        saved_img = []
        self.model.eval()

        # Get the shape for random noise generation 
        # DDPM only cares about channels, image_size can be random
        input_channels = getattr(self.model, 'input_channels', None) 
        if input_channels is None:
            raise AttributeError("Model does not have attribute 'input_channels'")
        
        to_pil = T.ToPILImage()

        # Start denoising 
        for i in range(num_images):
            z = torch.randn(1, input_channels, image_size, image_size).to(self.device)
            batch_size = z.shape[0]
            saved_imgs = []
            for t in reversed(range(1, self.num_steps)):
                temp = self.beta[t]/((torch.sqrt(1-self.alpha[t]))*(torch.sqrt(1-self.beta[t])))
                expanded_t = torch.tensor(t).repeat(batch_size).to(self.device)
                z = (1/(torch.sqrt(1-self.beta[t])))*z - (temp*self.model(z,expanded_t))

                if t in saving_steps:
                    img_tensor = z[:,:3,...].cpu().detach()
                    pil_img = to_pil(img_tensor[0])
                    saved_imgs.append(pil_img)
                
                e = torch.randn(1, input_channels, image_size, image_size).to(self.device)
                z = z + (e*torch.sqrt(self.beta[t]))
            
            # Final clean image
            temp = self.beta[0]/((torch.sqrt(1-self.alpha[0]))*(torch.sqrt(1-self.beta[0])))
            expanded_t0 = torch.tensor(0).repeat(batch_size).to(self.device)
            x = (1/(torch.sqrt(1-self.beta[0])))*z - (temp*self.model(z,expanded_t0))
            img_tensor = x[:,:3,...].cpu().detach()
            pil_img = to_pil(img_tensor[0])
            saved_imgs.append(pil_img)

            # Combine saved_imgs horizontally
            widths, heights = zip(*(img.size for img in saved_imgs))
            total_width = sum(widths)
            max_height = max(heights)
            combined_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in saved_imgs:
                combined_img.paste(img, (x_offset, 0))
                x_offset += img.width
            
            # Save combined image
            combined_img.save(os.path.join(self.outputs_dir, f'image_{i+1}.jpg'))

