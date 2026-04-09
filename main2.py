"""
Physics-Informed DeepONet 训练入口

使用方法:
    python main2.py  (根据 config.py 中的 use_parallel 自动选择单卡/多卡)
"""

import torch

from Labconfig import *
from config import *
from model.PI_DeepOnet import *
from model.plotting import *
from model.utils import get_available_gpus


def main():
    # 参数设置
    args = Args()

    # ==========================================
    # 检查训练模式
    # ==========================================
    use_parallel = getattr(args, 'use_parallel', False)

    if use_parallel:
        # 多 GPU 并行模式
        print("=" * 60)
        print("多 GPU 并行训练模式")
        print("=" * 60)

        # 检测可用 GPU
        num_gpus = getattr(args, 'num_gpus', 2)
        min_gpu_memory = getattr(args, 'min_gpu_memory', 10240)  # MB

        available_gpus = get_available_gpus(min_memory_mb=min_gpu_memory, require_count=num_gpus)

        if len(available_gpus) < num_gpus:
            print(f"⚠️ 可用 GPU 不足 ({len(available_gpus)} < {num_gpus})，回退到单 GPU 模式")
            args.use_parallel = False
            args.device = available_gpus[0] if available_gpus else 0
        else:
            print(f"✅ 检测到 {len(available_gpus)} 个可用 GPU: {available_gpus[:num_gpus]}")
            print("=" * 60)

            # 导入并调用分布式训练函数 (使用 mp.spawn 内部启动多进程)
            from model.train_distributed import train_distributed
            train_distributed(args)
            return

    # 单 GPU 模式
    print("=" * 60)
    print("单 GPU 训练模式")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(args.device)}")
        gpu_memory = torch.cuda.get_device_properties(args.device).total_memory / (1024 ** 3)
        print(f"GPU 内存: {gpu_memory:.1f} GB")
    else:
        print("⚠️ CUDA 不可用，将使用 CPU 训练")

    print("=" * 60)

    # 导入并调用单卡训练函数
    from model.train import train
    train(args)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print('*******************************************')
    print('           START TRAINING Pi_DeepONet      ')
    print('*******************************************')
    main()