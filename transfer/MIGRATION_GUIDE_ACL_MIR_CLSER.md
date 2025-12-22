# ACL、MIR、CLS-ER 方法迁移指南

## 📋 目录
1. [概述](#概述)
2. [迁移前准备](#迁移前准备)
3. [文件清单](#文件清单)
4. [详细迁移步骤](#详细迁移步骤)
5. [依赖关系说明](#依赖关系说明)
6. [配置参数](#配置参数)
7. [测试验证](#测试验证)
8. [常见问题](#常见问题)

---

## 概述

本指南用于将以下三个持续学习方法从当前代码库迁移到另一个相同架构的实验代码库：

- **ACL (Adaptive Continual Learning)**: 使用记忆回放、特征一致性和Hint蒸馏
- **CLS-ER (Complementary Learning System - Experience Replay)**: 使用双EMA模型和置信度选择
- **MIR (Maximally Interfered Retrieval)**: 基于梯度干扰的智能采样策略

### 方法特点对比

| 方法 | 核心创新 | 缓冲区类型 | 额外模型 | 计算开销 |
|------|---------|-----------|---------|---------|
| **ACL** | 双缓冲+特征蒸馏 | ReservoirBuffer + SoftBuffer | 1个教师模型 | 中等 |
| **CLS-ER** | 双EMA+置信度选择 | CLSER_Buffer | 2个EMA模型 | 中等 |
| **MIR** | 最大干扰采样 | MIR_Buffer | 虚拟模型（临时） | 较高 |

---

## 迁移前准备

### 1. 确认目标代码库架构

确保目标代码库具有以下结构：

```
TargetProject/
├── exp/
│   ├── exp_basic.py       # 基础实验类
│   ├── exp_main.py        # 主实验类
│   └── exp_online.py      # 在线学习实验类（需要存在或创建）
├── util/
│   ├── buffer.py          # 基础Buffer类（可选，会被ACL/MIR/CLSER独立使用）
│   └── metrics.py         # 评估指标
├── models/                # 模型定义
├── data_provider/         # 数据加载器
└── run.py                 # 主运行脚本
```

---

## 文件清单

### 必须迁移的文件

#### 1. 实验类文件（核心）
- **源文件**: `exp/exp_online.py`
- **需要迁移的部分**:
  - `Exp_ACL` 类 (L532-708)
  - `Exp_CLSER` 类 (L710-816)
  - `Exp_MIR` 类 (L818-939)

#### 2. 工具类文件
| 文件 | 功能 | 行数 | 依赖 |
|------|------|------|------|
| `util/acl_utils.py` | ACL专用工具（ReservoirBuffer, SoftBuffer） | 190 | torch, numpy, random |
| `util/clser_utils.py` | CLS-ER专用工具（CLSER_Manager, CLSER_Buffer） | 257 | torch, copy.deepcopy |
| `util/mir_utils.py` | MIR专用工具（MIR_Sampler, MIR_Buffer） | 335 | torch, numpy, copy.deepcopy |

### 可选参考文件（不需要迁移）
- `ACL/` - 原始参考实现
- `test_integration.py` - 集成测试脚本（可参考）
- `scripts/online/test_acl_methods.sh` - 测试脚本（可参考）

---

## 详细迁移步骤

### Step 1: 迁移工具类文件

#### 1.1 创建目标目录结构

```bash
cd /path/to/TargetProject
mkdir -p util
```

#### 1.2 复制工具类文件

```bash
# 复制三个工具类文件
cp /path/to/OnlineTSF/util/acl_utils.py ./util/
cp /path/to/OnlineTSF/util/clser_utils.py ./util/
cp /path/to/OnlineTSF/util/mir_utils.py ./util/
```

#### 1.3 验证工具类文件

确保每个文件开头的导入语句在目标环境中可用：

**acl_utils.py** 头部导入：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
```

**clser_utils.py** 头部导入：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
```

**mir_utils.py** 头部导入：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
```

---

### Step 2: 迁移实验类

#### 2.1 检查目标代码库的 exp_online.py

确认目标代码库中是否存在 `exp/exp_online.py`：

**如果存在**：跳到 Step 2.2

**如果不存在**：
1. 从源代码库复制基础的 `Exp_Online` 类
2. 或者创建一个继承自 `Exp_Main` 的基类

示例基础 `Exp_Online` 类：
```python
from exp.exp_main import Exp_Main

class Exp_Online(Exp_Main):
    def __init__(self, args):
        super().__init__(args)
        self.online_phases = ['test', 'online']

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        # 基础在线学习逻辑
        pass

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        # 基础在线更新逻辑
        pass
```

#### 2.2 添加三个实验类到 exp_online.py

在 `exp/exp_online.py` 文件末尾添加以下内容：

##### 2.2.1 添加 ACL 类

```python
# ============================================================================
# ACL (Adaptive Continual Learning) Methods
# ============================================================================

class Exp_ACL(Exp_Online):
    """
    ACL (Adaptive Continual Learning) Method

    核心创新：
    1. Memory Replay: 从长期记忆缓冲区重放样本
    2. Feature Consistency: 保持编码器特征的一致性
    3. Hint Distillation: 通过教师模型传递知识

    论文: "Adaptive Continual Learning for Time Series Forecasting"
    """
    def __init__(self, args):
        super().__init__(args)

        # ACL 超参数
        self.buffer_size = getattr(args, 'acl_buffer_size', 500)
        self.soft_buffer_size = getattr(args, 'acl_soft_buffer_size', 50)
        self.alpha = getattr(args, 'acl_alpha', 0.2)
        self.beta = getattr(args, 'acl_beta', 0.2)
        self.gamma = getattr(args, 'acl_gamma', 0.2)
        self.task_interval = getattr(args, 'acl_task_interval', 200)

        print(f"[ACL] Initialized with buffer_size={self.buffer_size}, "
              f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")

    # ... [复制完整的 Exp_ACL 实现，见源文件 L532-708] ...
```

**完整代码**：从源文件 `exp/exp_online.py` 的 **L532-708** 复制全部代码。

##### 2.2.2 添加 CLS-ER 类

```python
class Exp_CLSER(Exp_Online):
    """
    CLS-ER (Complementary Learning System - Experience Replay)

    核心创新：
    1. 双EMA模型：Plastic Model (快速学习) + Stable Model (稳定学习)
    2. 置信度选择：根据预测误差动态选择教师模型
    3. 一致性正则：学生模型与选中的教师保持一致

    论文: "Learning Fast, Learning Slow: A General Continual Learning Method
           based on Complementary Learning System" (ICLR 2022)
    """
    def __init__(self, args):
        super().__init__(args)

        # CLS-ER 超参数
        self.buffer_size = getattr(args, 'clser_buffer_size', 500)
        self.reg_weight = getattr(args, 'clser_reg_weight', 0.15)

        print(f"[CLS-ER] Initialized with buffer_size={self.buffer_size}, "
              f"reg_weight={self.reg_weight}")

    # ... [复制完整的 Exp_CLSER 实现，见源文件 L710-816] ...
```

**完整代码**：从源文件 `exp/exp_online.py` 的 **L710-816** 复制全部代码。

##### 2.2.3 添加 MIR 类

```python
class Exp_MIR(Exp_Online):
    """
    MIR (Maximally Interfered Retrieval)

    核心创新：
    - 不是随机采样buffer样本，而是选择受当前梯度更新负面影响最大的样本
    - 通过虚拟参数更新计算干扰分数
    - 选择top-K最大干扰样本进行回放

    论文: "Online Continual Learning with Maximal Interfered Retrieval" (NeurIPS 2019)
    """
    def __init__(self, args):
        super().__init__(args)

        # MIR 超参数
        self.buffer_size = getattr(args, 'mir_buffer_size', 500)
        self.mir_subsample = getattr(args, 'mir_subsample', 500)
        self.mir_k = getattr(args, 'mir_k', 50)

        print(f"[MIR] Initialized with buffer_size={self.buffer_size}, "
              f"subsample={self.mir_subsample}, k={self.mir_k}")

    # ... [复制完整的 Exp_MIR 实现，见源文件 L818-939] ...
```

**完整代码**：从源文件 `exp/exp_online.py` 的 **L818-939** 复制全部代码。

#### 2.3 添加必要的导入语句

在 `exp/exp_online.py` 文件开头添加：

```python
import copy
import torch
import torch.nn.functional as F
from torch import optim, nn
```

---

### Step 3: 修改主运行脚本

#### 3.1 更新 run.py 中的实验类映射

找到 `run.py` 中定义实验类的部分，添加三个新方法：

```python
# 在 run.py 中找到类似这样的代码：
if args.online_method == 'ER':
    Exp = Exp_ER
elif args.online_method == 'DER++':
    Exp = Exp_DERpp
# ... 其他方法 ...

# 添加以下三行：
elif args.online_method == 'ACL':
    Exp = Exp_ACL
elif args.online_method == 'CLSER':
    Exp = Exp_CLSER
elif args.online_method == 'MIR':
    Exp = Exp_MIR
```

或者使用更简洁的映射方式：

```python
from exp.exp_online import Exp_Online, Exp_ACL, Exp_CLSER, Exp_MIR

METHOD_MAP = {
    'online': Exp_Online,
    'ACL': Exp_ACL,
    'CLSER': Exp_CLSER,
    'MIR': Exp_MIR,
    # ... 其他方法 ...
}

Exp = METHOD_MAP.get(args.online_method, Exp_Online)
```

#### 3.2 添加命令行参数（如果使用 argparse）

在 `run.py` 或单独的参数配置文件中添加：

```python
# ACL 参数
parser.add_argument('--acl_buffer_size', type=int, default=500)
parser.add_argument('--acl_soft_buffer_size', type=int, default=50)
parser.add_argument('--acl_alpha', type=float, default=0.2, help='Memory replay weight')
parser.add_argument('--acl_beta', type=float, default=0.2, help='Feature consistency weight')
parser.add_argument('--acl_gamma', type=float, default=0.2, help='Hint distillation weight')
parser.add_argument('--acl_task_interval', type=int, default=200, help='Teacher update interval')

# CLS-ER 参数
parser.add_argument('--clser_buffer_size', type=int, default=500)
parser.add_argument('--clser_reg_weight', type=float, default=0.15, help='Consistency regularization weight')
parser.add_argument('--clser_plastic_update_freq', type=float, default=0.9)
parser.add_argument('--clser_plastic_alpha', type=float, default=0.999)
parser.add_argument('--clser_stable_update_freq', type=float, default=0.7)
parser.add_argument('--clser_stable_alpha', type=float, default=0.999)

# MIR 参数
parser.add_argument('--mir_buffer_size', type=int, default=500)
parser.add_argument('--mir_subsample', type=int, default=500, help='Subsample size for MIR')
parser.add_argument('--mir_k', type=int, default=50, help='Top-K interfered samples')
```

---

## 依赖关系说明

### 类继承关系

```
Exp_Basic (exp/exp_basic.py)
    ↓
Exp_Main (exp/exp_main.py)
    ↓
Exp_Online (exp/exp_online.py)
    ↓
    ├── Exp_ACL
    ├── Exp_CLSER
    └── Exp_MIR
```

### 方法依赖的父类接口

三个方法都依赖 `Exp_Online` 提供以下接口：

| 方法/属性 | 用途 | 必须存在 |
|----------|------|---------|
| `self.model` | 主模型 | ✅ |
| `self.device` | 计算设备 | ✅ |
| `self.args` | 超参数配置 | ✅ |
| `self._select_optimizer()` | 创建优化器 | ✅ |
| `self._select_criterion()` | 创建损失函数 | ✅ |
| `self.forward(batch)` | 前向传播 | ✅ |
| `super().online(...)` | 父类在线学习逻辑 | ✅ |

### 关键假设

1. **模型输出格式**：
   - 支持 `outputs` 或 `(outputs, encoder_features)` 两种格式
   - ACL 需要编码器特征用于 Hint Loss

2. **批次数据格式**：
   ```python
   batch = [batch_x, batch_y, batch_x_mark, batch_y_mark]
   # batch_x: [B, seq_len, enc_in]
   # batch_y: [B, pred_len, c_out]
   ```

3. **损失函数**：
   - 默认为 MSELoss
   - 支持自定义 criterion

---

## 配置参数

### ACL 推荐配置

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|-------|---------|------|
| `acl_buffer_size` | 500 | 200-1000 | 长期记忆容量 |
| `acl_soft_buffer_size` | 50 | 20-100 | 短期记忆容量 |
| `acl_alpha` | 0.2 | 0.1-0.5 | Memory replay权重 |
| `acl_beta` | 0.2 | 0.1-0.5 | Feature consistency权重 |
| `acl_gamma` | 0.2 | 0.1-0.5 | Hint distillation权重 |
| `acl_task_interval` | 200 | 100-500 | 教师模型更新间隔 |

**推荐组合**：
- 小数据集（<10K样本）：`buffer_size=200, alpha=0.1, beta=0.1, gamma=0.1`
- 中数据集（10K-100K）：默认参数
- 大数据集（>100K）：`buffer_size=1000, alpha=0.3, beta=0.3, gamma=0.3`

### CLS-ER 推荐配置

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|-------|---------|------|
| `clser_buffer_size` | 500 | 200-1000 | 缓冲区容量 |
| `clser_reg_weight` | 0.15 | 0.1-0.3 | 一致性正则权重 |
| `clser_plastic_update_freq` | 0.9 | 0.7-0.95 | Plastic模型更新频率 |
| `clser_plastic_alpha` | 0.999 | 0.99-0.9999 | Plastic EMA系数 |
| `clser_stable_update_freq` | 0.7 | 0.5-0.9 | Stable模型更新频率 |
| `clser_stable_alpha` | 0.999 | 0.99-0.9999 | Stable EMA系数 |

**关键原则**：
- `plastic_alpha < stable_alpha`（Plastic更新更快）
- `plastic_update_freq > stable_update_freq`（Plastic更新更频繁）

### MIR 推荐配置

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|-------|---------|------|
| `mir_buffer_size` | 500 | 200-1000 | 缓冲区容量 |
| `mir_subsample` | 500 | buffer_size的50%-100% | MIR候选样本数 |
| `mir_k` | 50 | 10-100 | Top-K干扰样本数 |

**性能权衡**：
- `mir_subsample` 越大，选择越准确，但计算开销越大
- 建议 `mir_k ≈ batch_size / 2`

---

## 测试验证

### Step 1: 单元测试工具类

创建 `test_utils.py`：

```python
import torch
from util.acl_utils import ReservoirBuffer, SoftBuffer
from util.clser_utils import CLSER_Manager, CLSER_Buffer
from util.mir_utils import MIR_Buffer

def test_acl_buffers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 ReservoirBuffer
    rb = ReservoirBuffer(capacity=100, device=device)
    x = torch.randn(8, 96, 7)
    y = torch.randn(8, 96, 7)
    z = torch.randn(8, 96, 64)
    rb.update(x, y, z)

    sampled = rb.sample(batch_size=4)
    assert sampled is not None
    print("✅ ReservoirBuffer test passed")

    # 测试 SoftBuffer
    sb = SoftBuffer(capacity=20, device=device)
    losses = torch.rand(8)
    sb.update(x, y, z, losses)
    data = sb.get_data()
    assert data is not None
    print("✅ SoftBuffer test passed")

def test_clser():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 CLSER_Buffer
    buffer = CLSER_Buffer(capacity=100, device=device)
    x = torch.randn(8, 96, 7)
    y = torch.randn(8, 96, 7)
    buffer.update(x, y)

    sampled_x, sampled_y, _ = buffer.sample(batch_size=4)
    assert sampled_x is not None
    print("✅ CLSER_Buffer test passed")

def test_mir():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建简单的 args
    class Args:
        mir_subsample = 50
        mir_k = 10
        learning_rate = 0.001

    args = Args()

    # 测试 MIR_Buffer
    buffer = MIR_Buffer(buffer_size=100, device=device, args=args)
    x = torch.randn(8, 96, 7)
    y = torch.randn(8, 96, 7)
    buffer.add_data(x, y)

    sampled = buffer.get_data(batch_size=4)
    assert sampled is not None
    print("✅ MIR_Buffer test passed")

if __name__ == '__main__':
    test_acl_buffers()
    test_clser()
    test_mir()
    print("\n🎉 All utility tests passed!")
```

运行测试：
```bash
cd /path/to/TargetProject
python test_utils.py
```

### Step 2: 集成测试

创建 `test_methods.sh`：

```bash
#!/bin/bash

# 测试 ACL
echo "Testing ACL..."
python run.py \
  --task_name long_term_forecast \
  --model DLinear \
  --data ETTh1 \
  --online_method ACL \
  --seq_len 96 \
  --pred_len 96 \
  --acl_buffer_size 100 \
  --train_epochs 1

# 测试 CLS-ER
echo "Testing CLS-ER..."
python run.py \
  --task_name long_term_forecast \
  --model DLinear \
  --data ETTh1 \
  --online_method CLSER \
  --seq_len 96 \
  --pred_len 96 \
  --clser_buffer_size 100 \
  --train_epochs 1

# 测试 MIR
echo "Testing MIR..."
python run.py \
  --task_name long_term_forecast \
  --model DLinear \
  --data ETTh1 \
  --online_method MIR \
  --seq_len 96 \
  --pred_len 96 \
  --mir_buffer_size 100 \
  --train_epochs 1

echo "✅ All integration tests completed!"
```

运行测试：
```bash
chmod +x test_methods.sh
./test_methods.sh
```

### Step 3: 性能对比测试

创建对比测试脚本：

```bash
#!/bin/bash

METHODS=("online" "ACL" "CLSER" "MIR")
RESULTS_DIR="./results_comparison"
mkdir -p $RESULTS_DIR

for method in "${METHODS[@]}"; do
    echo "Running $method..."
    python run.py \
        --online_method $method \
        --data ETTh1 \
        --seq_len 96 \
        --pred_len 96 \
        --train_epochs 5 \
        > ${RESULTS_DIR}/${method}_output.log 2>&1

    echo "$method completed. Results saved to ${RESULTS_DIR}/${method}_output.log"
done

echo "All methods completed. Compare results in $RESULTS_DIR/"
```

---

## 常见问题

### Q1: 迁移后报错 `ModuleNotFoundError: No module named 'util.acl_utils'`

**原因**: 工具类文件未正确复制或路径不对

**解决**:
```bash
# 检查文件是否存在
ls -la util/acl_utils.py
ls -la util/clser_utils.py
ls -la util/mir_utils.py

# 确保 util/ 是 Python 包
touch util/__init__.py
```

### Q2: 报错 `AttributeError: 'Namespace' object has no attribute 'acl_buffer_size'`

**原因**: 参数未在命令行或配置文件中定义

**解决**:
1. 检查 `run.py` 中是否添加了参数定义
2. 或者在调用时显式传递参数：
```bash
python run.py --online_method ACL --acl_buffer_size 500
```

### Q3: ACL 报错 `RuntimeError: encoder output is None`

**原因**: 模型不返回编码器特征，ACL需要encoder输出用于Hint Loss

**解决**:
- **方案1**: 修改模型使其返回 `(output, encoder_features)`
- **方案2**: 在 `Exp_ACL._update_online()` 中添加检查：
```python
if enc_out is None:
    # 跳过 Hint Loss
    loss_hint = torch.tensor(0.0, device=self.device)
```

### Q4: CLS-ER 显存占用过大

**原因**: 双EMA模型占用额外显存

**解决**:
1. 减小 `clser_buffer_size`
2. 使用混合精度训练：`--use_amp`
3. 减小模型尺寸

### Q5: MIR 运行速度很慢

**原因**: MIR需要额外的虚拟模型前向传播计算干扰分数

**解决**:
1. 减小 `mir_subsample`（减少候选样本）
2. 减小 `mir_k`（减少选择样本）
3. 增大 `batch_size`（摊销计算开销）

### Q6: 三个方法的效果不如预期

**原因**: 超参数未调优

**解决**:
参考 [配置参数](#配置参数) 章节，根据数据集特性调整：
- 小数据集：减小buffer大小和权重
- 大数据集：增大buffer大小和权重
- 数据分布变化快：增大 `task_interval`（ACL）或更新频率（CLS-ER）

### Q7: 如何与现有的在线学习方法（如ER, DER++）对比？

**答**: 在相同配置下运行：

```bash
# Baseline: 标准在线学习
python run.py --online_method online --data ETTh1

# ER (Experience Replay)
python run.py --online_method ER --data ETTh1

# ACL
python run.py --online_method ACL --data ETTh1

# CLS-ER
python run.py --online_method CLSER --data ETTh1

# MIR
python run.py --online_method MIR --data ETTh1
```

---

## 附录：完整迁移检查清单

### ✅ 迁移前检查
- [ ] 确认目标代码库架构与源代码库兼容
- [ ] 备份目标代码库
- [ ] 确认Python环境和依赖项

### ✅ 文件迁移
- [ ] 复制 `util/acl_utils.py`
- [ ] 复制 `util/clser_utils.py`
- [ ] 复制 `util/mir_utils.py`
- [ ] 在 `exp/exp_online.py` 中添加 `Exp_ACL` 类
- [ ] 在 `exp/exp_online.py` 中添加 `Exp_CLSER` 类
- [ ] 在 `exp/exp_online.py` 中添加 `Exp_MIR` 类

### ✅ 代码修改
- [ ] 更新 `run.py` 中的实验类映射
- [ ] 添加命令行参数定义
- [ ] 添加必要的导入语句

### ✅ 测试验证
- [ ] 运行工具类单元测试
- [ ] 运行ACL集成测试
- [ ] 运行CLS-ER集成测试
- [ ] 运行MIR集成测试
- [ ] 对比三个方法与baseline的性能

### ✅ 文档
- [ ] 在README中添加三个方法的说明
- [ ] 添加示例运行命令
- [ ] 记录超参数推荐配置

---

## 技术支持

如遇到迁移问题，请检查：

1. **日志输出**: 查看 `[ACL]`, `[CLS-ER]`, `[MIR]` 开头的日志
2. **源代码**: 参考 `OnlineTSF/exp/exp_online.py` L532-939
3. **参考实现**: 查看 `ACL/` 文件夹下的原始实现

---

## 版本历史

- **v1.0** (2025-01-XX): 初始版本
  - 支持 ACL, CLS-ER, MIR 三个方法的完整迁移

---

**祝迁移顺利！** 🚀
