"""
CLS-ER (Complementary Learning System - Experience Replay) Utilities
Adapted for Time Series Forecasting from the original image classification implementation.

Original Paper: "Learning Fast, Learning Slow: A General Continual Learning Method 
based on Complementary Learning System" (ICLR 2022)

Core Innovations:
1. Dual EMA Models: Plastic (fast learner) + Stable (slow learner)
2. Confidence-based Teacher Selection
3. Consistency Regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class CLSER_Manager:
    """
    CLS-ER 管理器：负责管理双 EMA 模型（Plastic + Stable）和一致性正则化
    
    Parameters:
    -----------
    model : nn.Module
        主模型（学生模型）
    args : Namespace
        包含超参数的配置对象
    device : torch.device
        计算设备
    """
    
    def __init__(self, model, args, device):
        self.model = model
        self.device = device
        self.global_step = 0
        
        # CLS-ER 超参数
        self.reg_weight = getattr(args, 'clser_reg_weight', 0.1)
        
        # Plastic Model 参数（快速学习）
        self.plastic_update_freq = getattr(args, 'clser_plastic_update_freq', 0.9)
        self.plastic_alpha = getattr(args, 'clser_plastic_alpha', 0.999)
        
        # Stable Model 参数（稳定学习）
        self.stable_update_freq = getattr(args, 'clser_stable_update_freq', 0.7)
        self.stable_alpha = getattr(args, 'clser_stable_alpha', 0.999)
        
        # 初始化两个 EMA 模型
        print("[CLS-ER] Initializing Plastic Model and Stable Model...")
        self.plastic_model = deepcopy(model).to(device)
        self.stable_model = deepcopy(model).to(device)
        
        # 设置为评估模式（不需要梯度）
        self.plastic_model.eval()
        self.stable_model.eval()
        for p in self.plastic_model.parameters():
            p.requires_grad = False
        for p in self.stable_model.parameters():
            p.requires_grad = False
            
        # 损失函数
        self.consistency_loss_fn = nn.MSELoss(reduction='none')
        
        print(f"[CLS-ER] Config: reg_weight={self.reg_weight}, "
              f"plastic(freq={self.plastic_update_freq}, alpha={self.plastic_alpha}), "
              f"stable(freq={self.stable_update_freq}, alpha={self.stable_alpha})")
    
    def compute_consistency_loss(self, buffer_batch):
        """
        计算一致性正则损失（核心创新）
        
        策略：
        1. 用 plastic 和 stable 模型分别预测 buffer 样本
        2. 计算每个样本的预测误差
        3. 选择误差小的作为教师（置信度选择）
        4. 主模型与选中教师的预测保持一致
        
        Parameters:
        -----------
        buffer_batch : list of torch.Tensor
            从 buffer 采样的批次 [batch_x, batch_y, ...]
            
        Returns:
        --------
        loss_consistency : torch.Tensor
            一致性正则损失
        """
        buffer_x = buffer_batch[0]
        buffer_y = buffer_batch[1]
        
        with torch.no_grad():
            # 1. 分别用两个 EMA 模型预测
            stable_pred = self.stable_model(buffer_x)
            plastic_pred = self.plastic_model(buffer_x)
            
            # 处理输出格式
            if isinstance(stable_pred, (tuple, list)):
                stable_pred = stable_pred[0]
            if isinstance(plastic_pred, (tuple, list)):
                plastic_pred = plastic_pred[0]
            
            # 2. 计算预测误差（替代原论文的 softmax 概率比较）
            # 误差越小 = 置信度越高
            stable_error = torch.mean((stable_pred - buffer_y) ** 2, dim=(1, 2))  # [B]
            plastic_error = torch.mean((plastic_pred - buffer_y) ** 2, dim=(1, 2)) # [B]
            
            # 3. 逐样本选择误差小的作为教师
            # sel_idx: [B, 1, 1] 用于广播
            sel_idx = (stable_error < plastic_error).float().view(-1, 1, 1)
            
            # 4. 混合教师预测（按样本选择）
            teacher_pred = sel_idx * stable_pred + (1 - sel_idx) * plastic_pred
        
        # 5. 主模型预测
        student_pred = self.model(buffer_x)
        if isinstance(student_pred, (tuple, list)):
            student_pred = student_pred[0]
        
        # 6. 计算一致性损失（MSE between student and teacher）
        loss_consistency = torch.mean(self.consistency_loss_fn(student_pred, teacher_pred.detach()))
        
        return loss_consistency
    
    def update_plastic_model(self):
        """
        更新 Plastic Model（指数移动平均 EMA）
        
        公式：θ_plastic = α * θ_plastic + (1-α) * θ_student
        其中 α 随训练步数动态调整
        """
        # 动态 alpha：初期更新快，后期更新慢
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_alpha)
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.plastic_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)
    
    def update_stable_model(self):
        """
        更新 Stable Model（指数移动平均 EMA）
        
        公式：θ_stable = α * θ_stable + (1-α) * θ_student
        Stable Model 的 α 更大，更新更慢
        """
        alpha = min(1 - 1 / (self.global_step + 1), self.stable_alpha)
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.stable_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)
    
    def maybe_update_ema_models(self):
        """
        按概率更新 EMA 模型（随机更新策略）
        
        为什么随机更新？
        - 避免每步都更新，减少计算开销
        - 增加训练的随机性和鲁棒性
        """
        self.global_step += 1
        
        # 随机更新 Plastic Model
        if torch.rand(1).item() < self.plastic_update_freq:
            self.update_plastic_model()
        
        # 随机更新 Stable Model
        if torch.rand(1).item() < self.stable_update_freq:
            self.update_stable_model()


class CLSER_Buffer:
    """
    CLS-ER 专用缓冲区（增强版 Reservoir Buffer）
    
    与原 ACL 的 ReservoirBuffer 的区别：
    - 存储完整的 (x, y, t_emb) 用于一致性损失计算
    - 支持更灵活的采样策略
    """
    
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.num_seen = 0
    
    def update(self, x, y, t_emb=None):
        """
        使用 Reservoir Sampling 策略更新缓冲区
        
        Parameters:
        -----------
        x : torch.Tensor [B, seq_len, enc_in]
        y : torch.Tensor [B, pred_len, c_out]
        t_emb : torch.Tensor [B, seq_len, time_dim], optional
        """
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            sample = {
                'x': x[i].detach().cpu(),
                'y': y[i].detach().cpu(),
                't_emb': t_emb[i].detach().cpu() if t_emb is not None else None
            }
            
            if len(self.buffer) < self.capacity:
                # 未满：直接添加
                self.buffer.append(sample)
            else:
                # 已满：按概率替换
                idx = torch.randint(0, self.num_seen + 1, (1,)).item()
                if idx < self.capacity:
                    self.buffer[idx] = sample
            
            self.num_seen += 1
    
    def sample(self, batch_size):
        """
        随机采样一个 mini-batch
        
        Returns:
        --------
        x, y, t_emb : torch.Tensor or None
        """
        if len(self.buffer) == 0:
            return None, None, None
        
        sample_size = min(batch_size, len(self.buffer))
        indices = torch.randperm(len(self.buffer))[:sample_size]
        
        batch_x = []
        batch_y = []
        batch_t_emb = []
        
        for idx in indices:
            sample = self.buffer[idx]
            batch_x.append(sample['x'])
            batch_y.append(sample['y'])
            if sample['t_emb'] is not None:
                batch_t_emb.append(sample['t_emb'])
        
        x = torch.stack(batch_x).to(self.device)
        y = torch.stack(batch_y).to(self.device)
        t_emb = torch.stack(batch_t_emb).to(self.device) if batch_t_emb else None
        
        return x, y, t_emb
    
    def is_empty(self):
        return len(self.buffer) == 0

