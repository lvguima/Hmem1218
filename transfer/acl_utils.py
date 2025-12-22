"""
ACL (Adaptive Continual Learning) Utilities for Time Series Forecasting

核心组件：
1. ReservoirBuffer: 长期记忆存储（Reservoir Sampling）
2. SoftBuffer: 短期记忆存储（Loss-based Selection）

注意：主项目已有util/buffer.py中的Buffer类，这里提供ACL专用的扩展版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class ReservoirBuffer:
    """
    标准内存缓冲区 (Standard Memory Buffer)
    使用蓄水池采样 (Reservoir Sampling) 维护长期记忆。
    
    存储元组: (x, y, z_hint, t_emb)
    - x: 输入序列
    - y: 目标输出
    - z_hint: 编码器特征（用于Hint Loss）
    - t_emb: 时间嵌入
    """
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.data = [] # List of dicts
        self.seen_samples = 0
    
    def update(self, x, y, z, t_emb=None):
        """
        将一个 Batch 的数据尝试加入 Buffer
        
        Parameters:
        -----------
        x : torch.Tensor [B, seq_len, enc_in]
        y : torch.Tensor [B, label_len+pred_len, c_out]
        z : torch.Tensor [B, seq_len, d_model] - 编码器特征
        t_emb : torch.Tensor [B, seq_len, time_dim], optional
        """
        batch_size = x.shape[0]
        
        # 必须 detach，否则会保存整个计算图导致内存爆炸
        x = x.detach().cpu()
        y = y.detach().cpu()
        z = z.detach().cpu()
        if t_emb is not None:
            t_emb = t_emb.detach().cpu()
            
        for i in range(batch_size):
            self.seen_samples += 1
            sample = {
                'x': x[i], 
                'y': y[i], 
                'z': z[i],
                't': t_emb[i] if t_emb is not None else None
            }
            
            if len(self.data) < self.capacity:
                # Buffer 未满，直接添加
                self.data.append(sample)
            else:
                # Buffer 已满，使用蓄水池采样概率替换
                idx = random.randint(0, self.seen_samples - 1)
                if idx < self.capacity:
                    self.data[idx] = sample
                    
    def sample(self, batch_size):
        """
        随机采样 Batch
        
        Returns:
        --------
        tuple: (batch_x, batch_y, batch_z, batch_t) or None
        """
        if len(self.data) == 0:
            return None
        
        sample_size = min(len(self.data), batch_size)
        samples = random.sample(self.data, sample_size)
        
        # 整理 batch
        batch_x = torch.stack([s['x'] for s in samples]).to(self.device)
        batch_y = torch.stack([s['y'] for s in samples]).to(self.device)
        batch_z = torch.stack([s['z'] for s in samples]).to(self.device)
        
        batch_t = None
        if samples[0]['t'] is not None:
            batch_t = torch.stack([s['t'] for s in samples]).to(self.device)
            
        return batch_x, batch_y, batch_z, batch_t

    def __len__(self):
        return len(self.data)


class SoftBuffer:
    """
    软缓冲区 (Soft Memory Buffer)
    
    用于存储当前 Batch 中 Loss 最小的样本（简单/一致样本），
    并在下一轮混合训练。
    
    每隔一定步数（task_interval）清空一次。
    
    核心思想：
    - 保留容易学习的样本（小loss）
    - 帮助模型保持对当前任务模式的短期记忆
    """
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = [] # Temporary storage
        
    def update(self, x, y, z, losses, t_emb=None):
        """
        根据 Loss 筛选样本存入
        
        Parameters:
        -----------
        x : torch.Tensor [B, seq_len, enc_in]
        y : torch.Tensor [B, label_len+pred_len, c_out]
        z : torch.Tensor [B, seq_len, d_model]
        losses : torch.Tensor [B] - 每个样本的 Loss 值
        t_emb : torch.Tensor [B, seq_len, time_dim], optional
        """
        x = x.detach().cpu()
        y = y.detach().cpu()
        z = z.detach().cpu()
        losses = losses.detach().cpu()
        if t_emb is not None:
            t_emb = t_emb.detach().cpu()
            
        # 找出 Loss 最小的 Top-K 索引
        # 注意：我们要找最小的 loss，即最容易学的/最符合当前模式的
        k = min(len(losses), self.capacity)
        if k == 0: 
            return
        
        # argsort 默认是从小到大，取前 k 个
        topk_indices = torch.argsort(losses)[:k]
        
        # 清空旧数据（Soft Buffer 只保留最新的"好"样本用于短期回顾）
        self.buffer = [] 
        
        for idx in topk_indices:
            sample = {
                'x': x[idx],
                'y': y[idx],
                'z': z[idx],
                't': t_emb[idx] if t_emb is not None else None
            }
            self.buffer.append(sample)
            
    def get_data(self):
        """
        取出所有数据用于训练
        
        Returns:
        --------
        tuple: (batch_x, batch_y, batch_z, batch_t) or None
        """
        if len(self.buffer) == 0:
            return None
            
        samples = self.buffer
        
        batch_x = torch.stack([s['x'] for s in samples]).to(self.device)
        batch_y = torch.stack([s['y'] for s in samples]).to(self.device)
        batch_z = torch.stack([s['z'] for s in samples]).to(self.device)
        
        batch_t = None
        if samples[0]['t'] is not None:
            batch_t = torch.stack([s['t'] for s in samples]).to(self.device)
            
        return batch_x, batch_y, batch_z, batch_t
    
    def clear(self):
        """清空缓冲区"""
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)

