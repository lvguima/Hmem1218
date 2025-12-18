"""
测试checkpoint路径生成逻辑
"""
import os
import sys

# 模拟args对象
class Args:
    def __init__(self):
        self.checkpoints = './checkpoints/'
        self.dataset = 'ETTh1'
        self.model = 'iTransformer'
        self.checkpoint_path = None

def resolve_checkpoint_path(args):
    """从run.py复制的函数"""
    base_dir = args.checkpoints if getattr(args, 'checkpoints', None) else './checkpoints/'
    # 按数据集/模型名组织checkpoint目录
    checkpoint_dir = os.path.join(base_dir, args.dataset, args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, 'checkpoint.pth')

def test_checkpoint_path():
    """测试不同配置下的checkpoint路径"""
    print("=" * 60)
    print("测试 Checkpoint 路径生成")
    print("=" * 60)

    # 测试1: 默认配置
    args1 = Args()
    path1 = resolve_checkpoint_path(args1)
    expected1 = os.path.join('./checkpoints/', 'ETTh1', 'iTransformer', 'checkpoint.pth')
    print(f"\n测试1 - 默认配置:")
    print(f"  数据集: {args1.dataset}")
    print(f"  模型: {args1.model}")
    print(f"  生成路径: {path1}")
    print(f"  期望路径: {expected1}")
    print(f"  [PASS]" if path1 == expected1 else f"  [FAIL]")

    # 测试2: 不同模型
    args2 = Args()
    args2.model = 'PatchTST'
    path2 = resolve_checkpoint_path(args2)
    expected2 = os.path.join('./checkpoints/', 'ETTh1', 'PatchTST', 'checkpoint.pth')
    print(f"\n测试2 - 不同模型:")
    print(f"  数据集: {args2.dataset}")
    print(f"  模型: {args2.model}")
    print(f"  生成路径: {path2}")
    print(f"  期望路径: {expected2}")
    print(f"  [PASS]" if path2 == expected2 else f"  [FAIL]")

    # 测试3: 不同数据集
    args3 = Args()
    args3.dataset = 'ETTh2'
    path3 = resolve_checkpoint_path(args3)
    expected3 = os.path.join('./checkpoints/', 'ETTh2', 'iTransformer', 'checkpoint.pth')
    print(f"\n测试3 - 不同数据集:")
    print(f"  数据集: {args3.dataset}")
    print(f"  模型: {args3.model}")
    print(f"  生成路径: {path3}")
    print(f"  期望路径: {expected3}")
    print(f"  [PASS]" if path3 == expected3 else f"  [FAIL]")

    # 测试4: 验证目录已创建
    print(f"\n测试4 - 验证目录创建:")
    print(f"  ./checkpoints/ETTh1/iTransformer 存在: {os.path.exists('./checkpoints/ETTh1/iTransformer')}")
    print(f"  ./checkpoints/ETTh1/PatchTST 存在: {os.path.exists('./checkpoints/ETTh1/PatchTST')}")
    print(f"  ./checkpoints/ETTh2/iTransformer 存在: {os.path.exists('./checkpoints/ETTh2/iTransformer')}")

    # 测试5: 验证不同模型不会互相覆盖
    print(f"\n测试5 - 路径隔离验证:")
    print(f"  ETTh1/iTransformer: {path1}")
    print(f"  ETTh1/PatchTST: {path2}")
    print(f"  路径不同: {'[PASS]' if path1 != path2 else '[FAIL]'}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

    # 清理测试创建的目录（可选）
    # import shutil
    # if os.path.exists('./checkpoints'):
    #     shutil.rmtree('./checkpoints')

if __name__ == '__main__':
    test_checkpoint_path()
