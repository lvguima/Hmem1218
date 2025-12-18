# 2024-12-14 修复与开发记录

5. 

---

## 2. run.py 默认参数修复

### 问题发现
- `--only_test` 默认值为 `True` (第39行)
- `--online_method` 默认值为 `'HMem'` (第51行)

这导致即使用户想要预训练，命令也会直接进入测试阶段。

### 修复内容

**文件**: `D:\pyproject\OnlineTSF-main\run.py`

```python
# 第51行: 将默认值从 'HMem' 改为 None
parser.add_argument('--online_method', type=str, default=None)

# 第491行: 添加 show_progress=True 以显示进度
mse, mae, test_data = exp.online(test_data, show_progress=True)
```

---

## 3. H-Mem NaN Loss 问题修复

### 问题现象
```
[Step 500] Loss: nan | Running MSE: nan
```

### 根本原因
1. 梯度爆炸
2. 神经记忆更新中的数值不稳定
3. 内存扩展时传播旧的 NaN 值

### 修复文件和内容

#### 文件1: `exp/exp_hmem.py`

**`_update_online` 方法修复**:

```python
def _update_online(self, batch, criterion, optimizer, scaler=None, **kwargs) -> float:
    # ... 设置代码 ...

    # 检查输入中的 NaN
    if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
        print("[Warning] NaN detected in input batch, skipping update")
        return 0.0

    try:
        # 前向传播带 NaN 检查
        if torch.isnan(prediction).any():
            print("[Warning] NaN detected in prediction, skipping update")
            if hasattr(model, 'snma'):
                model.snma.reset(batch_size=batch_x.size(0))
            return 0.0

        # Loss NaN 检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("[Warning] NaN/Inf detected in loss, skipping update")
            if hasattr(model, 'snma'):
                model.snma.reset(batch_size=batch_x.size(0))
            return 0.0

        # 始终裁剪梯度
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    except RuntimeError as e:
        print(f"[Warning] RuntimeError in update: {e}")
        if hasattr(model, 'snma'):
            model.snma.reset(batch_size=batch_x.size(0))
        return 0.0
```

**`online()` 方法 - 添加进度打印**:

```python
# 进度打印间隔
print_interval = 500

# 在循环中:
if (i + 1) % print_interval == 0 and self.args.local_rank <= 0:
    avg_loss = np.mean(losses[-print_interval:]) if losses else 0
    recent_preds = np.concatenate(preds[-print_interval:], axis=0)
    recent_trues = np.concatenate(trues[-print_interval:], axis=0)
    running_mse = np.mean((recent_preds - recent_trues) ** 2)
    print(f"  [Step {i+1}] Loss: {avg_loss:.6f} | Running MSE: {running_mse:.6f}")

# 修复返回格式:
return mse, mae, online_data  # 原来是: return mse, preds, trues
```

#### 文件2: `adapter/module/neural_memory.py`

**`_expand_memory_for_batch` 方法修复**:

```python
def _expand_memory_for_batch(self, batch_size: int):
    if self.memory.size(0) != batch_size:
        # 重置为零以避免 NaN 传播
        device = self.memory.device
        self.memory = torch.zeros(batch_size, self.memory_dim, device=device)
        self.memory_age = torch.zeros(batch_size, device=device)
```

**`update` 方法修复**:

```python
def update(self, new_info, surprise):
    # 将 surprise 限制在有效范围
    surprise = torch.clamp(surprise, 0.0, 1.0)

    # 检查输入中的 NaN
    if torch.isnan(new_info).any() or torch.isnan(surprise).any():
        return self.memory

    # ... 记忆更新逻辑 ...

    # 限制值以防止极端情况
    M_new = torch.clamp(M_new, -10.0, 10.0)

    # 检查 NaN 并回退
    if torch.isnan(M_new).any():
        M_new = M_prev

    return M_new
```

---

## 4. 公平对比评估分析

### 确认公平性的关键点

1. **相同数据划分**: 所有方法使用相同的 train/val/test 划分 (settings.py `get_borders()`)
2. **相同预训练模型**: 所有在线学习方法从相同的 checkpoint 开始
3. **相同在线学习流程**: update → predict 循环
4. **无信息泄漏**: 仅使用历史数据进行更新
5. **相同评估指标**: MSE, MAE

---

## 5. 常用命令

### 预训练 iTransformer
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 3 --batch_size 32 --learning_rate 0.0001
```

### H-Mem 快速测试
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.0001 --lora_rank 2 --lora_alpha 4.0 --memory_dim 64 --bottleneck_dim 8 --memory_capacity 50 --retrieval_top_k 2 --hmem_warmup_steps 5 --freeze True --use_snma True --use_chrc False
```

### ER 方法测试
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ER --only_test --pretrain --online_learning_rate 0.0001 --freeze False
```

---

## 6. 修改的文件清单

| 文件路径 | 修改内容 |
|---------|---------|
| `run.py` | 修改 `--online_method` 默认值，添加进度显示 |
| `exp/exp_hmem.py` | NaN 保护、梯度裁剪、进度打印、返回值修复 |
| `adapter/module/neural_memory.py` | 内存扩展重置、NaN 检查、值限制 |

