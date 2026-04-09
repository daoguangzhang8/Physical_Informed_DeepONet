# 单机多卡并行训练使用指南

## 1. 配置参数说明 (config.py)

```python
# ==========================================
# 2.1 单机多卡并行训练配置 (Single-Machine Multi-GPU)
# ==========================================
use_parallel = False                      # 是否启用多 GPU 并行训练 (True: 多GPU | False: 单GPU)
num_gpus = 2                              # 使用的 GPU 数量
min_gpu_memory = 10 * 1024                # GPU 最小可用内存 (MB)，低于此值的 GPU 不会被使用
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_parallel` | bool | False | True: 多GPU并行 / False: 单GPU训练 |
| `num_gpus` | int | 2 | 使用的 GPU 数量 |
| `min_gpu_memory` | int | 10240 | GPU 最小可用内存 (MB) |

## 2. 使用方法

### 单 GPU 训练

```python
# config.py
use_parallel = False
device = 3  # 指定使用的 GPU 编号
```

```bash
python main2.py
```

### 单机多 GPU 并行训练

```python
# config.py
use_parallel = True
num_gpus = 2
min_gpu_memory = 10 * 1024  # 10GB
```

```bash
python main2.py  # 自动使用 mp.spawn 启动多进程
```

## 3. 工作原理

程序启动时会自动检测 `use_parallel` 配置：

```
main2.py
    │
    ├── use_parallel = False ──→ model/train.py::train()
    │
    └── use_parallel = True  ──→ model/train_distributed.py::train_distributed()
                                        │
                                        └── mp.spawn() 启动 N 个进程
```

**关键**: 使用 `torch.multiprocessing.spawn` 内部启动多进程，无需手动使用 `torchrun`

## 4. 工具函数说明

| 函数 | 用途 |
|------|------|
| `setup_distributed(rank, world_size)` | 初始化单机多卡环境 |
| `cleanup_distributed()` | 清理分布式环境 |
| `get_available_gpus(min_memory_mb, require_count)` | 获取满足内存要求的 GPU 列表 |
| `wrap_model_for_distributed(model, rank)` | 将模型包装为 DDP 模型 |
| `is_main_process(rank)` | 检查是否为主进程 |
| `reduce_tensor(tensor, op)` | 跨进程归约张量 |

## 5. 注意事项

1. **启动方式**: 只需 `python main2.py`，程序内部自动启动多进程
2. **GPU 内存检测**: 程序启动时会自动检测可用 GPU，低于 `min_gpu_memory` 的 GPU 会被排除
3. **Batch Size**: 使用多 GPU 时，实际 batch size = `batch_size * num_gpus`
4. **学习率**: 可考虑根据 GPU 数量线性缩放学习率
5. **模型保存**: 只在主进程 (rank 0) 保存模型权重和生成图表
6. **数据加载**: 使用 `DistributedSampler` 确保数据不重复
7. **DDP 模型访问**: 分布式模式下，访问原始模型需使用 `model.module`

## 6. 性能优化建议

- 适当增加 `num_workers` 以加速数据加载
- 使用混合精度训练 (`torch.cuda.amp`) 进一步提升性能
- 确保 GPU 间通信带宽充足 (使用 NVLink 或 PCIe 4.0)

## 7. 常见问题

### Q1: 如何选择 GPU 数量?
A: 在 `config.py` 中设置 `num_gpus`，程序会自动检测并选择满足内存要求的 GPU。

### Q2: 出现 "CUDA out of memory" 怎么办?
A: 尝试以下方法:
- 减小 `batch_size` 或 `batch_size_v`
- 增大 `min_gpu_memory` 排除内存不足的 GPU
- 减少 `num_gpus` 使用更少的 GPU

### Q3: 如何加载 DDP 保存的模型?
A: DDP 保存的是 `model.module.state_dict()`，加载时:
```python
# 单 GPU 推理
model.load_state_dict(torch.load('model.pth')['model_state_dict'])

# 继续分布式训练
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[rank])
model.load_state_dict(torch.load('model.pth')['model_state_dict'])
```
