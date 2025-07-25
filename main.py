#!/usr/bin/env python3
"""
CLIPN Medical Domain Training Script
===================================

This script provides a complete training setup for CLIPN adaptation to medical domain
using BiomedCLIP and ROCOv2 dataset with proper experiment tracking and configuration management.

Usage:
    python main.py --config configs/example_config.yaml
    python main.py --config configs/example_config.yaml --experiment-name "my_experiment"
    python main.py --config configs/example_config.yaml --resume checkpoints/latest.pth
"""

import argparse
import os
import sys
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
import io
import ast
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import datasets

# Import your custom modules
from src.clipn import AltCLIPNAdapter
from src.utils import ROCOv2CaptionsDataset
from src.training import CLIPNLoss
from open_clip import create_model_from_pretrained, get_tokenizer

@dataclass
class TrainingConfig:
    """Configuration dataclass for training hyperparameters"""
    # Basic training parameters
    epochs: int = 55
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Advanced training settings
    gradient_accumulation_steps: int = 16
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    max_grad_norm: float = 1.0
    
    # Model configuration
    model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    transformer_path: str = "text.transformer.encoder"
    num_no_texts: int = 16
    frozen_visual: bool = True
    
    # Loss configuration
    temperature: float = 0.07
    loss_mode: str = "cosine"  # "cosine" or "L2"
    
    # Data configuration
    dataset_name: str = "axiong/pmc_oa"
    dataset_config: str = "pmc_oa"
    dataset_split: str = "train"
    streaming: bool = False
    num_workers: int = 1
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

def setup_logging(experiment_name: str, log_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    # Create log directory
    log_path = Path(log_dir) / experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    
    return logger

def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create TrainingConfig instance with loaded parameters
    config = TrainingConfig(**config_dict)
    
    return config

def save_config(config: TrainingConfig, save_path: str):
    """Save configuration to file"""
    config_dict = {k: v for k, v in config.__dict__.items()}
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def create_model_and_tokenizer(config: TrainingConfig, logger: logging.Logger):
    """Create CLIPN model and tokenizer"""
    logger.info(f"Loading base model: {config.model_name}")
    
    # Load base model and tokenizer
    base_model, preprocess = create_model_from_pretrained(config.model_name)
    tokenizer = get_tokenizer(config.model_name)
    
    # Create CLIPN adapter
    clipn_model = AltCLIPNAdapter(
        base_model,
        tokenizer,
        num_no_texts=config.num_no_texts,
        transformer_path=config.transformer_path,
        frozen=config.frozen_visual
    )
    
    logger.info(f"CLIPN model created with {sum(p.numel() for p in clipn_model.parameters())} parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in clipn_model.parameters() if p.requires_grad)}")
    
    return clipn_model, tokenizer, preprocess

def create_data_loader(config: TrainingConfig, tokenizer, preprocess, logger: logging.Logger):
    """Create data loader for training"""
    logger.info(f"Preparing {config.dataset_name} dataset")

    def preprocess_samples(some_samples):
        imgs = []
        for img in some_samples["image"]:
            img_bytes = ast.literal_eval(img)["bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes))
            imgs.append(preprocess(pil_img))

        caption = tokenizer(some_samples["caption"])
        return {
            "images": imgs,
            "captions": caption
        }
    
    # Create dataset
    dataset = datasets.load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split
    ).with_transform(preprocess_samples)

    def collate_fn(batch):
        images = [item["images"] for item in batch]
        captions = [item["captions"] for item in batch]
        images = torch.stack(images)
        captions = torch.stack(captions)
        return images, captions
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info(f"Data loader created with batch size {config.batch_size}")
    
    return data_loader

def create_optimizer_and_scheduler(model, config: TrainingConfig, total_steps: int, logger: logging.Logger):
    """Create optimizer and learning rate scheduler"""
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Create learning rate scheduler
    if config.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            # T_max=total_steps - config.warmup_steps,
            T_max = config.warmup_steps * 5,
            eta_min=config.learning_rate * 0.01
        )
    elif config.scheduler_type == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
    else:  # constant
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    # Warmup scheduler
    if config.warmup_steps > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
        
        logger.info(f"Created warmup scheduler for {config.warmup_steps} steps")
    else:
        warmup_scheduler = None
    
    logger.info(f"Created optimizer: AdamW with LR={config.learning_rate}")
    logger.info(f"Created scheduler: {config.scheduler_type}")
    
    return optimizer, scheduler, warmup_scheduler

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir: str, experiment_name: str, logger: logging.Logger):
    """Save model checkpoint"""
    checkpoint_path = Path(checkpoint_dir) / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save latest checkpoint
    latest_path = checkpoint_path / 'latest_checkpoint.pth'
    torch.save(checkpoint, latest_path)
    
    # Save epoch-specific checkpoint
    epoch_path = checkpoint_path / f'checkpoint_epoch_{epoch}_step_{step}.pth'
    torch.save(checkpoint, epoch_path)
    
    logger.info(f"Checkpoint saved: {latest_path}")

def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler, logger: logging.Logger):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0, 0
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    
    logger.info(f"Checkpoint loaded: epoch {epoch}, step {step}")
    
    return epoch, step

def train_epoch(model, data_loader, optimizer, scheduler, warmup_scheduler, loss_fn, 
                config: TrainingConfig, writer: SummaryWriter, epoch: int, 
                global_step: int, logger: logging.Logger):
    """Train for one epoch"""
    model.train()
    
    epoch_losses = []
    batch_times = []
    
    # Progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for batch_idx, (images, captions) in enumerate(pbar):
        batch_start_time = time.time()
        
        # Move to device
        images = images.to(config.device)
        captions = captions.to(config.device)
        captions = captions.reshape((images.shape[0], -1))
        
        # Forward pass
        image_features, text_features, text_features_no, logit_scale_exp = model(images, captions)
        
        # Compute loss
        total_loss, loss_dict = loss_fn(image_features, text_features, text_features_no)
        
        # Scale loss for gradient accumulation
        loss = total_loss / config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Scheduler step
            if warmup_scheduler is not None and global_step < config.warmup_steps:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            global_step += 1
        
        # Logging
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        epoch_losses.append(loss_dict)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to TensorBoard
        if batch_idx % config.log_interval == 0:
            writer.add_scalar('Training/Total_Loss', loss_dict['total_loss'], global_step)
            writer.add_scalar('Training/ITBO_Loss', loss_dict['itbo_loss'], global_step)
            writer.add_scalar('Training/TSO_Loss', loss_dict['tso_loss'], global_step)
            writer.add_scalar('Training/Learning_Rate', current_lr, global_step)
            writer.add_scalar('Training/Batch_Time', batch_time, global_step)
            
            # Log gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
        
        # Update progress bar
        pbar.set_postfix({
            'Total Loss': f"{loss_dict['total_loss']:.4f}",
            'ITBO': f"{loss_dict['itbo_loss']:.4f}",
            'TSO': f"{loss_dict['tso_loss']:.4f}",
            'LR': f"{current_lr:.2e}",
            'Step': global_step
        })
        
        # Save checkpoint
        if global_step % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                loss_dict['total_loss'], config.checkpoint_dir,
                writer.log_dir.split('/')[-1], logger
            )
    
    # Epoch summary
    avg_loss = np.mean([l['total_loss'] for l in epoch_losses])
    avg_itbo = np.mean([l['itbo_loss'] for l in epoch_losses])
    avg_tso = np.mean([l['tso_loss'] for l in epoch_losses])
    avg_batch_time = np.mean(batch_times)
    
    logger.info(f"Epoch {epoch+1} Summary:")
    logger.info(f"  Average Total Loss: {avg_loss:.4f}")
    logger.info(f"  Average ITBO Loss: {avg_itbo:.4f}")
    logger.info(f"  Average TSO Loss: {avg_tso:.4f}")
    logger.info(f"  Average Batch Time: {avg_batch_time:.2f}s")
    logger.info(f"  Learning Rate: {current_lr:.2e}")
    
    # Log epoch summary to TensorBoard
    writer.add_scalar('Epoch/Total_Loss', avg_loss, epoch)
    writer.add_scalar('Epoch/ITBO_Loss', avg_itbo, epoch)
    writer.add_scalar('Epoch/TSO_Loss', avg_tso, epoch)
    writer.add_scalar('Epoch/Batch_Time', avg_batch_time, epoch)
    
    return global_step

def evaluate_model(model, data_loader, loss_fn, config: TrainingConfig, logger: logging.Logger):
    """Evaluate model on validation set"""
    model.eval()
    
    total_similarities = []
    eval_losses = []
    
    with torch.no_grad():
        for images, captions in tqdm(data_loader, desc="Evaluating"):
            images = images.to(config.device)
            captions = captions.to(config.device)
            captions = captions.reshape((images.shape[0], -1))
            
            # Forward pass
            image_features, text_features, text_features_no, _ = model(images, captions)
            
            # Compute loss
            total_loss, loss_dict = loss_fn(image_features, text_features, text_features_no)
            eval_losses.append(loss_dict)
            
            # Compute similarities
            similarities = F.cosine_similarity(image_features, text_features, dim=-1)
            total_similarities.extend(similarities.cpu().numpy())
    
    avg_similarity = np.mean(total_similarities)
    avg_loss = np.mean([l['total_loss'] for l in eval_losses])
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Average Cosine Similarity: {avg_similarity:.4f}")
    logger.info(f"  Average Loss: {avg_loss:.4f}")
    
    return avg_similarity, avg_loss

def main():
    """Main training function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CLIPN Medical Domain Training')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment (default: timestamp)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (reduces data and epochs)')
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"clipn_medical_{timestamp}"
    
    # Load configuration
    config = load_config(args.config)
    
    # Debug mode adjustments
    if args.debug:
        config.epochs = 2
        config.batch_size = 8
        config.log_interval = 5
        config.save_interval = 50
        args.experiment_name = f"debug_{args.experiment_name}"
    
    # Setup logging
    logger = setup_logging(args.experiment_name, config.log_dir)
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Configuration loaded from: {args.config}")
    
    # Create experiment directory
    experiment_dir = Path(config.log_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, experiment_dir / 'config.yaml')
    
    # Log system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create model and tokenizer
    model, tokenizer, preprocess = create_model_and_tokenizer(config, logger)
    model = model.to(config.device)
    
    # Create data loader
    data_loader = create_data_loader(config, tokenizer, preprocess, logger)
    
    # Calculate total steps
    total_steps = len(data_loader) * config.epochs // config.gradient_accumulation_steps
    logger.info(f"Total training steps: {total_steps}")
    
    # Create optimizer and scheduler
    optimizer, scheduler, warmup_scheduler = create_optimizer_and_scheduler(
        model, config, total_steps, logger
    )
    
    # Create loss function
    loss_fn = CLIPNLoss(temperature=config.temperature, mode=config.loss_mode)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=str(experiment_dir))
    
    # Log configuration to TensorBoard
    config_text = yaml.dump(config.__dict__, default_flow_style=False)
    writer.add_text('Config', config_text, 0)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler, logger
        )
    
    # Training loop
    logger.info("Starting training...")
    best_similarity = 0.0
    
    try:
        for epoch in range(start_epoch, config.epochs):
            logger.info(f"Starting epoch {epoch+1}/{config.epochs}")
            
            # Train for one epoch
            global_step = train_epoch(
                model, data_loader, optimizer, scheduler, warmup_scheduler,
                loss_fn, config, writer, epoch, global_step, logger
            )
            
            # Evaluate model (on a subset for efficiency)
            eval_loader = DataLoader(
                data_loader.dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )
            
            # Limit evaluation to first 100 batches
            eval_samples = min(100, len(eval_loader))
            eval_subset = torch.utils.data.Subset(
                eval_loader.dataset, 
                range(0, eval_samples * config.batch_size)
            )
            eval_loader = DataLoader(eval_subset, batch_size=config.batch_size, shuffle=False)
            
            avg_similarity, avg_loss = evaluate_model(model, eval_loader, loss_fn, config, logger)
            
            # Log evaluation results
            writer.add_scalar('Evaluation/Cosine_Similarity', avg_similarity, epoch)
            writer.add_scalar('Evaluation/Loss', avg_loss, epoch)
            
            # Save best model
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    avg_loss, config.checkpoint_dir, f"{args.experiment_name}_best", logger
                )
                logger.info(f"New best model saved with similarity: {best_similarity:.4f}")
            
            # Save regular checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                avg_loss, config.checkpoint_dir, args.experiment_name, logger
            )
            
            logger.info(f"Epoch {epoch+1} completed. Best similarity so far: {best_similarity:.4f}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            0.0, config.checkpoint_dir, f"{args.experiment_name}_interrupted", logger
        )
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        writer.close()
        logger.info("Training completed")

if __name__ == "__main__":
    main()
