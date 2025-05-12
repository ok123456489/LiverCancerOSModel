import os
import torch
import logging
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader

from model import SurvivalModel
from utils.matrics import Evaluate


class SurvivalTrainer:
    def __init__(self, model, train_data, config):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.early_stop = config.get('early_stop', 10)
        self.train_loader = DataLoader(
            train_data,
            batch_size=config['batch_size'],
            shuffle=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=1e-4
        )
        self.config = config
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: max(0.95 ** epoch, 5e-6 / config['lr']))
        # 确保输出目录存在
        self.output_dir = config.get('output_dir', './output')
        os.makedirs(self.output_dir, exist_ok=True)

        # 生成唯一的模型保存路径
        self.model_id = config.get('model_id', 'model')
        self.best_model_path = os.path.join(
            self.output_dir,
            f"best_{self.model_id}.pth"
        )

    def train_epoch(self):
        epoch_loss = 0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        self.scheduler.step()
        return epoch_loss / len(self.train_loader)

    def run_training(self):
        metrics_log = {
            'train_loss': [],
            'cindex': [],
        }
        best_score = -np.inf
        epochs_no_improve = 0
        self.model.train()
        for epoch in range(self.config['epochs']):
            epoch_loss = self.train_epoch()
            metrics_log['train_loss'].append(epoch_loss)
            metrics = Evaluate(self.model, self.train_loader)
            metrics_log['cindex'].append(metrics['c_index'])

            print(f"Epoch {epoch + 1}: loss={epoch_loss:.4f}, C-index={metrics['c_index']:.3f}")


            if metrics['c_index'] > best_score:
                best_score = metrics['c_index']
                model_config = {
                    'slice_fusion_type': getattr(self.model, 'slice_fusion_type', self.model.slice_fusion_type),
                    'seq_fusion_type': getattr(self.model, 'seq_fusion_type', self.model.seq_fusion_type),
                    'use_clinical': getattr(self.model, 'use_clinical', self.model.use_clinical),
                    'clinical_dim': getattr(self.model, 'clinical_dim', 41),
                    'use_SNN': getattr(self.model, 'use_SNN', True)
                }
                # 保存完整模型信息
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': epoch_loss,
                    'cindex': metrics['c_index'],
                    'model_config': model_config,
                    'train_config': self.config
                }, self.best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                # if epochs_no_improve == self.early_stop:
                #     print(f"Early stopping at epoch {epoch}")
                #     break

        from visualization import plot_training_metrics  # 导入绘图函数
        plot_path = os.path.join(self.output_dir, f"training_metrics_{self.model_id}.png")
        plot_training_metrics(metrics_log, save_path=plot_path)
        return metrics_log

    @staticmethod
    def load_model(checkpoint_path, device=None, strict=False):
        """加载保存的模型"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 重建模型
        model = SurvivalModel(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model.to(device)
        model.eval()

        return {
            'model': model,
            'epoch': checkpoint['epoch'],
            'optimizer_state': checkpoint['optimizer_state_dict'],
            'loss': checkpoint['loss'],
            'cindex': checkpoint['cindex'],
            'config': checkpoint['train_config']
        }