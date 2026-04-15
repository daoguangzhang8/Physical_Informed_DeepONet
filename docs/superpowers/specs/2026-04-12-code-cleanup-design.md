# PI-DeepONet 代码清理与可读性优化设计

> 日期：2026-04-12
> 原则：不更改任何计算逻辑，只清理死代码、修复拼写、统一 import、补充 .gitignore

## 范围

本文档覆盖两类安全优化：

1. **代码可读性与死代码清理**（6 个子项）
2. **.gitignore 补充**（1 项）

所有修改严格保证：不影响训练/推理的数值结果、不改变计算逻辑、不调整运行时行为。

---

## 1. 消除 `from Labconfig import *` 通配符污染

### 问题

每个文件都使用 `from Labconfig import *`，导致命名空间被 ~30 个不需要的符号污染，IDE 无法正确补全和类型检查。

### 修改方案

每个文件替换为显式导入。逐文件映射如下：

**`model/PI_DeepOnet.py`：**
```python
# 替换 from Labconfig import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
```

**`model/net_module.py`：**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
```

**`model/utils.py`：**
```python
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import qmc
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
```

**`model/dataloader.py`：**
```python
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.utils import Halton_Sample
```

**`model/train.py`：**
```python
import os
import gc
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from model.utils import WarmupScheduler, count_parameters
from model.dataloader import prepare_training_dataloaders, prepare_external_val_dataset
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.plotting import plot_loss, test_plot, plot_sinlge
```

**`model/train_distributed.py`：**
```python
import os
import gc
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.utils import (
    setup_distributed, cleanup_distributed, wrap_model_for_distributed,
    is_main_process, reduce_tensor, count_parameters, WarmupScheduler
)
from model.dataloader import prepare_training_dataloaders, prepare_external_val_dataset
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.plotting import plot_loss, test_plot, plot_sinlge, fine_tuning
```

**`model/plotting.py`：**
```python
import os
import copy
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from config import Args
from model.utils import calculate_regression_metrics
```

**`main2.py`：**
```python
import torch
from config import Args
from model.utils import get_available_gpus
```
注意：`main2.py` 中 `from model.PI_DeepOnet import *` 和 `from model.plotting import *` 也是通配符，但实际上 main2.py 并未直接使用这些模块中的任何符号（只是导入了），可以删除这两行。

**`test.py`：**
保持显式导入不变（已经是显式导入）。

**`model/FNO.py`：**
```python
import torch
import torch.nn as nn
from model.net_module import FNO2d, PositionalEncoding, FourierFeatureEncoder
```
注意：如果清理死代码后 FNO 不再使用 `PositionalEncoding` 和 `FourierFeatureEncoder`，则只导入 `FNO2d`。

**`Labconfig.py`：** 保留原样（它本身就是 import hub），但在文件顶部添加注释说明仅作为集中导入使用。

---

## 2. 删除 `net_module.py` 中未使用的网络组件

### 待删除列表

| 组件 | 行数 | 原因 |
|------|------|------|
| `QR_orthogonalization()` 函数 | ~205-239 | 全项目零调用 |
| `StandardCrossAttention` 类 | ~329-385 | 全项目零调用 |
| `StandardEncoderLayer` 类 | ~387-440 | 全项目零调用 |
| `Tokenizer` 类 | ~442-458 | 全项目零调用 |
| `FourierFeatureEncoder` 类 | ~304-327 | 仅 FNO.py 定义但 forward 未使用 |
| `PositionalEembedding` 类 | ~96-111 | 拼写错误类名，零调用 |
| `ResidualBlock` 类 | ~39-52 | 全项目零调用 |

### 验证方法

修改前后运行 `grep -r "QR_orthogonalization\|StandardCrossAttention\|StandardEncoderLayer\|Tokenizer\|FourierFeatureEncoder\|PositionalEembedding\|ResidualBlock" --include="*.py"` 确认无引用。

---

## 3. 清理 `FNO.py` 中未使用的模型参数

### 待删除属性

```python
# 以下属性在 forward() 中均未使用
self.pos_encoder = ...        # 删除
self.fencoder = ...           # 删除
self.pos_scale = ...          # 删除
self.data_norm_coe = 1.       # 删除
self.pde_norm_coe = 1.        # 删除
self.pde_real_k = 0.          # 删除
self.pde_imag_k = 0.          # 删除
self.log_var_data = ...       # 删除
self.log_var_pde = ...        # 删除
self.feat_dim = 256           # 删除
self.b2 = ...                 # 删除
self.device = ...             # 删除
self.encoded_dim = 4          # 删除
```

### 保留

```python
self.FNO = nn.Sequential(...)  # 唯一在 forward 中使用的模块
```

### 清理后 FNO.py

```python
import torch.nn as nn
from model.net_module import FNO2d

class FNO(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_shape_branch1 = args.input_shape_branch1
        input_shape_branch2 = args.input_shape_branch2
        self.FNO = nn.Sequential(
            FNO2d(input_shape_branch1[1] + input_shape_branch2[1], 2, 16, 16, 128),
        )

    def forward(self, vel, UU0):
        fin = torch.cat([vel, UU0], dim=1)
        return self.FNO(fin)
```

---

## 4. 清理 `PI_DeepOnet.py` 中未使用的属性和方法

### 待删除属性

| 属性 | 原因 |
|------|------|
| `self.log_var_data` | loss() 和 compute_loss() 中未使用 |
| `self.log_var_pde` | loss() 和 compute_loss() 中未使用 |
| `self.block_feature_encoder` | forward() 中注释标记为未使用 |

### 待删除方法

| 方法 | 原因 |
|------|------|
| `loss_BC()` | 训练/测试流程中零调用 |
| `loss_PDE_Scatter_pml()` | 训练/测试流程中零调用（已被 `_compute_pde_residual` 替代） |
| `loss_Reg()` | 训练/测试流程中零调用 |
| `envelope_barrier_loss()` | 训练/测试流程中零调用 |
| `get_trunk_output()` | 训练/测试流程中零调用 |
| `dynamic_barrier_loss()` | 训练/测试流程中零调用 |
| `get_ortho_loss()` | 训练/测试流程中零调用 |

### 验证

搜索确认：`grep -r "loss_BC\|loss_PDE_Scatter_pml\|loss_Reg\|envelope_barrier_loss\|get_trunk_output\|dynamic_barrier_loss\|get_ortho_loss" --include="*.py"`

---

## 5. 修复拼写错误

| 位置 | 修改 |
|------|------|
| `model/plotting.py` 函数名 | `plot_sinlge` → `plot_single` |
| 所有调用处 | `plot_sinlge(` → `plot_single(` |

涉及文件：`plotting.py`（定义）、`train.py`、`train_distributed.py`

---

## 6. 清理注释掉的代码块

### `train.py`

删除以下注释块（约 20 行）：
- L222-224: marmousi_data 相关注释
- L228-229: 微调 test_plot 注释
- L233: Marmousi test_plot 注释

### `PI_DeepOnet.py`

删除以下注释行：
- L29: `# self.fencoder = FourierFeatureEncoder(...)` — 未使用注释
- L55: `# self.loss_function_point = ...` — 未使用注释

### `dataloader.py`

删除文件末尾大段注释掉的 `extract_single_model_multi_source` 函数（L331-372）。

### `test.py`

删除注释掉的测试数据集配置（如 `'1994BP'`, `'SEAM'`）。

---

## 7. `.gitignore` 补充

在现有 `.gitignore` 中添加：

```gitignore
# Runtime artifacts
nohup.out
*.out

# Python cache
__pycache__/
*.pyc
*.pyo

# Data files (typically large)
*.npy
*.npz

# Model checkpoints (large binary)
*.pth
*.pt

# OS artifacts
.DS_Store
```

---

## 不在范围内

以下内容经用户确认不做修改：

- 计算冗余消除（Part 2 全部）
- 验证路径 `torch.no_grad()` 优化（Part 4）
- 重复函数/代码合并（test.py 中的重复实现）
- 重复 import 合并（test.py）
- 运行效率优化（AMP、向量化循环等）
