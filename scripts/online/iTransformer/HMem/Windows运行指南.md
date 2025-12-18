# Windows 下运行 H-Mem 实验指南

由于 bash 脚本在 Windows 下可能遇到 conda 环境传递问题，我们提供了三种运行方式：

## 方法 1：使用 Python 脚本（推荐 ✅）

这是最可靠的跨平台方式：

```bash
# 1. 激活 conda 环境
conda activate cl

# 2. 进入项目根目录
cd D:\pyproject\OnlineTSF-main

# 3. 运行 Python 脚本
python scripts/online/iTransformer/HMem/quick_test.py
```

**优点**：
- 直接使用当前激活的 Python 环境
- 跨平台兼容（Windows/Linux/Mac）
- 实时输出结果到控制台和日志文件

---

## 方法 2：使用 Windows 批处理脚本

在 Anaconda Prompt 中运行：

```cmd
# 打开 Anaconda Prompt（开始菜单搜索）
# 进入项目目录
cd D:\pyproject\OnlineTSF-main

# 运行批处理脚本
scripts\online\iTransformer\HMem\quick_test.bat
```

**优点**：
- 自动激活 conda 环境
- 自动验证依赖
- Windows 原生支持

---

## 方法 3：直接运行 Python 命令

最简单的方式，适合调试：

```bash
# 1. 激活环境
conda activate cl

# 2. 进入项目目录
cd D:\pyproject\OnlineTSF-main

# 3. 直接运行（复制以下完整命令）
python -u run.py \
  --dataset ETTh2 --border_type online \
  --model iTransformer \
  --seq_len 96 \
  --pred_len 24 \
  --online_method HMem \
  --itr 1 \
  --only_test \
  --learning_rate 0.0001 \
  --online_learning_rate 0.0001 \
  --lora_rank 4 \
  --lora_alpha 8.0 \
  --memory_dim 128 \
  --bottleneck_dim 16 \
  --memory_capacity 100 \
  --retrieval_top_k 3 \
  --pogt_ratio 0.5 \
  --hmem_warmup_steps 10 \
  --freeze True \
  --use_snma True \
  --use_chrc True
```

**注意**：在 Windows cmd 中，需要把 `\` 改成 `^` 或者写成一行。

---

## 运行其他数据集

### 使用 Python 脚本运行完整实验

创建一个运行脚本 `run_hmem_etth2.py`：

```python
import sys
import subprocess

# H-Mem 配置
config = {
    'dataset': 'ETTh2',
    'model': 'iTransformer',
    'online_method': 'HMem',
    'seq_len': 336,
    'itr': 3,  # 3次迭代

    # H-Mem 超参数
    'lora_rank': 8,
    'lora_alpha': 16.0,
    'memory_dim': 256,
    'bottleneck_dim': 32,
    'memory_capacity': 1000,
    'retrieval_top_k': 5,
    'pogt_ratio': 0.5,
    'hmem_warmup_steps': 100,

    'learning_rate': 0.0001,
    'online_learning_rate': 0.0001,
}

# 运行 3 个预测视野
for pred_len in [24, 48, 96]:
    print(f"\n{'='*60}")
    print(f"Running {config['dataset']} with horizon={pred_len}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-u", "run.py",
        "--dataset", config['dataset'],
        "--border_type", "online",
        "--model", config['model'],
        "--seq_len", str(config['seq_len']),
        "--pred_len", str(pred_len),
        "--online_method", config['online_method'],
        "--itr", str(config['itr']),
        "--only_test",
        "--learning_rate", str(config['learning_rate']),
        "--online_learning_rate", str(config['online_learning_rate']),
        "--lora_rank", str(config['lora_rank']),
        "--lora_alpha", str(config['lora_alpha']),
        "--memory_dim", str(config['memory_dim']),
        "--bottleneck_dim", str(config['bottleneck_dim']),
        "--memory_capacity", str(config['memory_capacity']),
        "--retrieval_top_k", str(config['retrieval_top_k']),
        "--pogt_ratio", str(config['pogt_ratio']),
        "--hmem_warmup_steps", str(config['hmem_warmup_steps']),
        "--freeze", "True",
        "--use_snma", "True",
        "--use_chrc", "True",
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nError running pred_len={pred_len}")
        sys.exit(1)

print(f"\n{'='*60}")
print("All experiments completed!")
print(f"{'='*60}\n")
```

然后运行：
```bash
conda activate cl
cd D:\pyproject\OnlineTSF-main
python run_hmem_etth2.py
```

---

## 故障排除

### 问题 1：找不到 numpy

**症状**：`ModuleNotFoundError: No module named 'numpy'`

**解决**：
```bash
# 确认当前环境
conda activate cl
python -c "import numpy; print(numpy.__version__)"

# 如果失败，重新安装
pip install numpy pandas torch
```

### 问题 2：找不到 checkpoint

**症状**：`FileNotFoundError: checkpoint not found`

**解决**：
确保已经预训练过模型：
```bash
python run.py --model iTransformer --dataset ETTh2 --seq_len 96 --pred_len 24
```

或者下载预训练好的 checkpoint。

### 问题 3：CUDA out of memory

**解决**：
减小批次大小或内存维度：
```bash
python run.py ... --batch_size 8 --memory_dim 128 --memory_capacity 500
```

---

## 验证安装

快速验证 H-Mem 模块是否正确安装：

```bash
conda activate cl
cd D:\pyproject\OnlineTSF-main

# 测试导入
python -c "from adapter.hmem import HMem; print('✓ H-Mem import successful')"
python -c "from exp.exp_hmem import Exp_HMem; print('✓ Exp_HMem import successful')"

# 运行单元测试
python -m pytest tests/test_hmem_integration.py -v
```

---

## 推荐运行顺序

1. **快速验证**（5-10分钟）：
   ```bash
   python scripts/online/iTransformer/HMem/quick_test.py
   ```

2. **单个数据集完整实验**（2-3小时）：
   ```bash
   # 修改 quick_test.py，设置 pred_len=[24,48,96], itr=3
   ```

3. **消融实验**（评估组件贡献）

4. **超参数调优**（找到最佳配置）

5. **全数据集对比**（最终评估）

---

**注意**：
- 始终在 `conda activate cl` 激活环境后运行
- 确保在项目根目录 `D:\pyproject\OnlineTSF-main` 下执行
- 使用 Python 脚本比 bash 脚本更可靠
