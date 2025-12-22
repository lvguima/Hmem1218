"""
MIR (Maximally Interfered Retrieval) Utilities for Time Series Forecasting
Adapted from NeurIPS 2019 paper: "Online Continual Learning with Maximal Interfered Retrieval"

Core Innovation:
Instead of random sampling from buffer, MIR selects samples that will be 
MOST NEGATIVELY AFFECTED by the current gradient update.

Key Idea:
1. Compute current gradient
2. Simulate parameter update (virtual model)
3. Calculate loss increase for each buffer sample
4. Select samples with highest interference scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np


def get_grad_vector(model):
    """
    收集模型所有梯度到一个向量中
    
    Parameters:
    -----------
    model : nn.Module
        当前模型
        
    Returns:
    --------
    grads : torch.Tensor [total_params]
        梯度向量
    grad_dims : list
        每层参数数量列表
    """
    grad_dims = []
    for param in model.parameters():
        grad_dims.append(param.data.numel())
    
    grads = torch.zeros(sum(grad_dims))
    if next(model.parameters()).is_cuda:
        grads = grads.cuda()
    
    cnt = 0
    for param in model.parameters():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    
    return grads, grad_dims


def overwrite_grad(model, new_grad, grad_dims):
    """
    用新的梯度向量覆盖模型梯度
    
    Parameters:
    -----------
    model : nn.Module
    new_grad : torch.Tensor
        新梯度向量
    grad_dims : list
        每层参数数量
    """
    cnt = 0
    for param in model.parameters():
        param.grad = torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1


def get_future_step_parameters(model, grad_vector, grad_dims, lr=1.0):
    """
    模拟梯度更新后的虚拟模型参数
    
    公式: θ_virtual = θ_current - lr * gradient
    
    Parameters:
    -----------
    model : nn.Module
        当前模型
    grad_vector : torch.Tensor
        梯度向量
    grad_dims : list
        每层参数数量
    lr : float
        学习率
        
    Returns:
    --------
    virtual_model : nn.Module
        虚拟更新后的模型（深拷贝）
    """
    virtual_model = deepcopy(model)
    overwrite_grad(virtual_model, grad_vector, grad_dims)
    
    with torch.no_grad():
        for param in virtual_model.parameters():
            if param.grad is not None:
                param.data = param.data - lr * param.grad.data
    
    return virtual_model


class MIR_Sampler:
    """
    MIR 采样器：选择最大干扰样本
    
    核心算法：
    1. 计算当前批次梯度
    2. 模拟参数更新（虚拟模型）
    3. 在 buffer 样本上计算更新前后的损失
    4. interference_score = loss_after - loss_before
    5. 选择 top-K 最高分数样本
    """
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # MIR 超参数
        self.subsample = getattr(args, 'mir_subsample', 500)  # 从 buffer 采样多少样本计算干扰
        self.k = getattr(args, 'mir_k', 50)  # 选择 top-K 样本
        
        print(f"[MIR] Sampler initialized: subsample={self.subsample}, k={self.k}")
    
    def compute_interference_scores(self, model, buffer_batch, grad_vector, grad_dims, criterion):
        """
        计算 buffer 样本的干扰分数
        
        Parameters:
        -----------
        model : nn.Module
            当前模型
        buffer_batch : list of torch.Tensor
            Buffer 批次数据 [batch_x, batch_y, ...]
        grad_vector : torch.Tensor
            当前梯度
        grad_dims : list
            参数维度
        criterion : nn.Module
            损失函数
            
        Returns:
        --------
        interference_scores : torch.Tensor [N]
            每个样本的干扰分数（越高越重要）
        """
        # 解包batch数据
        batch_x = buffer_batch[0]
        batch_y = buffer_batch[1]
        
        with torch.no_grad():
            # 1. 当前模型预测（更新前）
            outputs = model(batch_x)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            
            # 逐样本计算损失（不要 reduction）
            loss_pre = torch.mean((outputs - batch_y) ** 2, dim=(1, 2))  # [N]
        
        # 2. 模拟参数更新（虚拟模型）
        virtual_model = get_future_step_parameters(
            model, grad_vector, grad_dims, lr=self.args.learning_rate
        )
        virtual_model.eval()
        
        with torch.no_grad():
            # 3. 虚拟模型预测（更新后）
            outputs_post = virtual_model(batch_x)
            if isinstance(outputs_post, (tuple, list)):
                outputs_post = outputs_post[0]
            
            # 逐样本计算损失
            loss_post = torch.mean((outputs_post - batch_y) ** 2, dim=(1, 2))  # [N]
        
        # 4. 计算干扰分数：更新后损失 - 更新前损失
        # 正值 = 损失增加 = 受负面影响 = 需要回放
        interference_scores = loss_post - loss_pre  # [N]
        
        del virtual_model  # 释放内存
        
        return interference_scores
    
    def select_samples(self, model, buffer, criterion, batch_size):
        """
        使用 MIR 策略从 buffer 选择样本
        
        Returns:
        --------
        selected_batch : list of torch.Tensor
            选择的最大干扰样本批次
        """
        # 1. 从 buffer 采样候选样本（避免计算所有样本）
        buffer_size = min(self.subsample, buffer.num_seen_examples, len(buffer.buffer[0]))
        
        if buffer_size == 0:
            return None
        
        # 随机采样候选
        indices = np.random.choice(
            min(buffer.num_seen_examples, len(buffer.buffer[0])),
            size=buffer_size, 
            replace=False
        )
        
        # 获取候选样本
        buffer_batch = []
        for attr in buffer.buffer:
            buffer_batch.append(attr[indices].to(self.device))
        
        # 2. 获取当前梯度
        # 注意：在调用此函数前，当前批次的损失已经 backward 了
        grad_vector, grad_dims = get_grad_vector(model)
        
        # 3. 计算干扰分数
        scores = self.compute_interference_scores(
            model, buffer_batch, grad_vector, grad_dims, criterion
        )
        
        # 4. 选择 top-K 最大干扰样本
        k = min(batch_size, len(scores))
        _, top_indices = torch.topk(scores, k, largest=True)
        
        # 5. 返回选中的样本
        selected_batch = []
        for attr_batch in buffer_batch:
            selected_batch.append(attr_batch[top_indices])
        
        return selected_batch


class MIR_Buffer:
    """
    MIR 专用 Buffer（继承自 Reservoir Sampling）
    
    与标准 Buffer 的区别：
    - 支持 MIR 采样策略
    - 兼容主项目的 Buffer 接口
    """
    
    def __init__(self, buffer_size, device, args):
        self.buffer_size = buffer_size
        self.device = device
        self.args = args
        self.buffer = []
        self.num_seen_examples = 0
        
        # MIR 采样器
        self.mir_sampler = MIR_Sampler(args, device)
    
    def init_tensors(self, *batch):
        """初始化 buffer tensors"""
        for attr in batch:
            self.buffer.append(
                torch.zeros((self.buffer_size, *attr.shape[1:]), 
                           dtype=torch.float32, device=self.device)
            )
    
    def add_data(self, *batch):
        """
        使用 Reservoir Sampling 更新 buffer
        
        Parameters:
        -----------
        batch : tuple of torch.Tensor
            (batch_x, batch_y, ...)
        """
        if self.num_seen_examples == 0:
            self.init_tensors(*batch)
        
        for i in range(batch[0].shape[0]):
            # Reservoir sampling
            if self.num_seen_examples < self.buffer_size:
                index = self.num_seen_examples
            else:
                index = np.random.randint(0, self.num_seen_examples + 1)
            
            self.num_seen_examples += 1
            
            if index < self.buffer_size:
                for j, attr in enumerate(batch):
                    self.buffer[j][index] = attr[i].detach().to(self.device)
    
    def get_data_with_mir(self, model, criterion, batch_size):
        """
        使用 MIR 策略采样
        
        Returns:
        --------
        selected_batch : list of torch.Tensor or None
        """
        if self.num_seen_examples == 0:
            return None
        
        return self.mir_sampler.select_samples(
            model, self, criterion, batch_size
        )
    
    def get_data(self, batch_size):
        """
        随机采样（作为对比baseline）
        
        Returns:
        --------
        batch : list of torch.Tensor or None
        """
        if self.num_seen_examples == 0:
            return None
        
        sample_size = min(batch_size, self.num_seen_examples, len(self.buffer[0]))
        indices = np.random.choice(
            min(self.num_seen_examples, len(self.buffer[0])),
            size=sample_size,
            replace=False
        )
        
        batch = []
        for attr in self.buffer:
            batch.append(attr[indices].to(self.device))
        
        return batch
    
    def is_empty(self):
        return self.num_seen_examples == 0

