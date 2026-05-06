# 基于每个 Epoch 全训练集 Velocity 平均结构图的共享 `y_ran` 采样修改计划

## 0. 修改目标

当前由于显存限制，同一个 batch 内所有速度模型共享同一组 `y_ran` 采样点。原始 per-model 自适应结构采样虽然更精确，但会带来更高显存和计算压力。

本计划的目标是将 `y_ran` 改成：

```text
每个 epoch 开始时：
    基于所有训练 velocity model 计算一个全局/epoch 级结构概率图

每个 velocity batch 训练时：
    从这个 epoch 级概率图中采样一组共享 y_ran
    再 expand 到 batch 内所有速度模型
```

也就是从：

```text
per-model adaptive sampling:
    P_b(z, x) ∝ |∇v_b(z, x)|
```

改成：

```text
epoch-level shared sampling:
    P_epoch(z, x) ∝ mean over all training velocities of |∇v(z, x)|
```

如果希望更激进，也可以使用：

```text
P_epoch(z, x) = 0.7 * mean(|∇v|) + 0.3 * max(|∇v|)
```

但本计划优先实现“对所有 velocity model 进行平均采样”的版本。

---

## 1. 为什么采用 epoch-level 全局平均采样

### 1.1 当前 batch 共享采样的问题

由于同一个 batch 内所有速度模型共享同一组 `y_ran`，如果采样点只根据某一个速度模型生成，例如：

```python
y_ran = generate_structure_aware_y_ran(vel_batch[0])
```

那么采样点会偏向该速度模型的结构位置，对其他速度模型不公平。

如果对 batch 内所有模型做平均：

```text
P_batch(z, x) = mean_b |∇v_b(z, x)|
```

则只代表当前 batch，而不同 batch 的采样分布会变化较大。

### 1.2 epoch-level 全局平均采样的优势

使用所有训练速度模型构造：

```text
P_epoch(z, x) = mean_all_train_vel |∇v(z, x)|
```

优点：

```text
1. 不偏向某一个 batch 或某一个 velocity model；
2. 每个 batch 使用相同的全局结构先验，训练更稳定；
3. 采样逻辑简单，显存友好；
4. 每个 epoch 可重新采样 y_ran，但概率图稳定；
5. 适合当前 batch 内共享 y_ran 的显存限制；
6. 采样开销很低，主要计算只在 epoch 开始前完成一次。
```

### 1.3 可能的缺点

```text
1. 如果不同速度模型的结构位置差异很大，mean 图会被平滑；
2. 对每个单独 velocity 的结构自适应能力弱于 per-model 采样；
3. 可能过度采样“全局常见结构”，漏掉少数特殊模型结构；
4. 需要配合 uniform / surface / source-near 点，避免采样过度集中。
```

---

## 2. 推荐采样组成

建议先不要增加 `num_pts`，保持当前：

```text
num_pts = 900
```

只改变采样点的组成。

### 2.1 无 source-near 的第一版

如果暂时不加入震源附近采样，推荐：

```text
60% epoch-structure
20% surface
20% uniform
```

对应 `num_pts=900`：

```text
epoch-structure: 540
surface:         180
uniform:         180
```

### 2.2 加入 source-near 的高频增强版

如果能获得固定震源坐标，推荐：

```text
45% epoch-structure
20% surface
20% source-near
15% uniform
```

对应 `num_pts=900`：

```text
epoch-structure: 405
surface:         180
source-near:     180
uniform:         135
```

对于高频单频模型和多震源任务，建议优先使用这个版本。

---

## 3. 修改总体流程

### 3.1 原始流程

当前类似：

```python
with torch.no_grad():
    y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)

for batch in dataloader['train_y']:
    loss = model.loss(..., y_ran=y_ran)
```

其中 `y_ran` 依赖当前 `vel_batch`。

### 3.2 修改后流程

改为：

```python
for epoch in range(NIter):

    # Step 1: epoch 开始时，基于所有训练 velocity 计算全局结构概率图
    epoch_prob = build_epoch_velocity_gradient_prob(dataloader['train'])

    for vel_batch in dataloader['train']:

        # Step 2: 从 epoch_prob 采样一组共享 y_ran
        y_shared = sample_shared_y_ran_from_epoch_prob(epoch_prob, num_pts=900)

        # Step 3: expand 到当前 batch
        y_ran = y_shared.unsqueeze(0).expand(B_v, -1, -1).clone()

        # Step 4: 进入 loss 时 requires_grad
        y_ran.requires_grad_(True)

        loss = model.loss(..., y_ran=y_ran)
```

### 3.3 更高效的变体

如果每个 epoch 都重新计算全局概率图过慢，可以先固定若干 epoch 更新一次：

```text
每 1 个 epoch 更新一次：最准确
每 5 个 epoch 更新一次：更省
只在训练开始前计算一次：最快
```

因为 velocity 数据本身不变，`P_epoch` 理论上可以只计算一次。

如果训练集中每个阶段数据不变，推荐先使用：

```text
每个 stage 开始前计算一次 P_stage
```

而不是每个 epoch 重算。

但如果你希望严格执行“每个 epoch 对所有 vel model 进行平均采样”，可以先按每个 epoch 实现，确认无误后再优化为缓存版本。

---

## 4. 新增函数一：计算 epoch-level velocity gradient 概率图

### 4.1 函数目标

输入训练 dataloader，遍历所有训练 velocity model，计算平均速度梯度图：

```text
score(z, x) = mean_i |∇v_i(z, x)|
```

输出归一化概率分布：

```text
prob: [Z * X]
```

### 4.2 推荐函数接口

```python
def build_epoch_velocity_gradient_prob(
    model,
    train_loader,
    device,
    eps=1e-8,
    use_max_mix=False,
    mean_weight=0.7,
    max_weight=0.3,
):
    ...
```

### 4.3 实现代码

```python
import torch
import torch.nn.functional as F

def build_epoch_velocity_gradient_prob(
    model,
    train_loader,
    device,
    eps=1e-8,
    use_max_mix=False,
    mean_weight=0.7,
    max_weight=0.3,
):
    """
    基于整个训练集 velocity model 构造 epoch-level 结构采样概率图。
    """
    score_sum = None
    score_max_global = None
    count = 0

    with torch.no_grad():
        for batch_data in train_loader:
            vel_batch = batch_data[0]
            vel = vel_batch.to(device)  # [B, 1, Z, X]
            B, _, Z, X = vel.shape

            grad_z = vel[:, :, 2:, 1:-1] - vel[:, :, :-2, 1:-1]
            grad_x = vel[:, :, 1:-1, 2:] - vel[:, :, 1:-1, :-2]

            grad_mag = torch.sqrt(grad_z ** 2 + grad_x ** 2 + eps)
            grad_mag = F.pad(
                grad_mag,
                (1, 1, 1, 1),
                mode="replicate"
            ).squeeze(1)  # [B, Z, X]

            batch_sum = grad_mag.sum(dim=0)  # [Z, X]

            if score_sum is None:
                score_sum = torch.zeros_like(batch_sum)
                score_max_global = torch.zeros_like(batch_sum)

            score_sum += batch_sum
            count += B

            if use_max_mix:
                batch_max = grad_mag.max(dim=0).values
                score_max_global = torch.maximum(score_max_global, batch_max)

    score_mean = score_sum / max(count, 1)

    if use_max_mix:
        score = mean_weight * score_mean + max_weight * score_max_global
    else:
        score = score_mean

    score = torch.clamp(score, min=0.0)
    prob = score.reshape(-1)
    prob = prob / (prob.sum() + eps)

    return prob, score
```

### 4.4 说明

默认建议先用：

```python
use_max_mix = False
```

也就是纯平均：

```text
P_epoch ∝ mean(|∇v|)
```

如果发现平均图过于平滑，再试：

```python
use_max_mix = True
```

即：

```text
P_epoch = 0.7 * mean(|∇v|) + 0.3 * max(|∇v|)
```

---

## 5. 新增函数二：从 epoch_prob 采样共享 `y_ran`

### 5.1 函数目标

输入：

```text
epoch_prob: [Z * X]
```

输出：

```text
y_shared: [num_pts, 2]
```

其中坐标格式为：

```text
[z, x]，物理坐标
```

### 5.2 推荐函数接口

```python
def sample_shared_y_ran_from_epoch_prob(
    prob,
    args,
    num_pts=900,
    structure_ratio=0.60,
    surface_ratio=0.20,
    uniform_ratio=0.20,
    source_ratio=0.0,
    source_coords=None,
    surface_depth_grids=5,
    source_r_min_grids=1.5,
    source_r_max_grids=8.0,
    replacement=True,
):
    ...
```

### 5.3 实现代码

```python
def sample_shared_y_ran_from_epoch_prob(
    prob,
    args,
    num_pts=900,
    structure_ratio=0.60,
    surface_ratio=0.20,
    uniform_ratio=0.20,
    source_ratio=0.0,
    source_coords=None,
    surface_depth_grids=5,
    source_r_min_grids=1.5,
    source_r_max_grids=8.0,
    replacement=True,
):
    """
    从 epoch-level 概率图中采样一组共享 y_ran。
    """
    device = prob.device
    nz, nx, dh = args.nz, args.nx, args.dh

    max_z = nz * dh
    max_x = nx * dh

    if source_coords is None or source_ratio <= 0:
        uniform_ratio = uniform_ratio + source_ratio
        source_ratio = 0.0

    num_structure = int(num_pts * structure_ratio)
    num_surface = int(num_pts * surface_ratio)
    num_source = int(num_pts * source_ratio)
    num_uniform = num_pts - num_structure - num_surface - num_source

    y_parts = []

    # 1. epoch-level structure points
    if num_structure > 0:
        sampled_indices = torch.multinomial(
            prob,
            num_samples=num_structure,
            replacement=replacement
        )

        z_idx = sampled_indices // nx
        x_idx = sampled_indices % nx

        z = z_idx.float() * dh + torch.rand(num_structure, device=device) * dh
        x = x_idx.float() * dh + torch.rand(num_structure, device=device) * dh

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_struct = torch.stack([z, x], dim=-1)
        y_parts.append(y_struct)

    # 2. surface points
    if num_surface > 0:
        surface_depth = surface_depth_grids * dh

        z = torch.rand(num_surface, device=device) * surface_depth
        x = torch.rand(num_surface, device=device) * max_x

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_surface = torch.stack([z, x], dim=-1)
        y_parts.append(y_surface)

    # 3. source-near points
    if num_source > 0 and source_coords is not None:
        source_coords = source_coords.to(device)

        src_id = torch.randint(
            low=0,
            high=source_coords.shape[0],
            size=(num_source,),
            device=device
        )

        src = source_coords[src_id]

        theta = 2.0 * torch.pi * torch.rand(num_source, device=device)

        r_min = source_r_min_grids * dh
        r_max = source_r_max_grids * dh
        r = r_min + (r_max - r_min) * torch.rand(num_source, device=device)

        z = src[:, 0] + r * torch.cos(theta)
        x = src[:, 1] + r * torch.sin(theta)

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_source = torch.stack([z, x], dim=-1)
        y_parts.append(y_source)

    # 4. uniform points
    if num_uniform > 0:
        z = torch.rand(num_uniform, device=device) * max_z
        x = torch.rand(num_uniform, device=device) * max_x

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_uniform = torch.stack([z, x], dim=-1)
        y_parts.append(y_uniform)

    y_shared = torch.cat(y_parts, dim=0)

    # 修正四舍五入造成的点数误差
    if y_shared.shape[0] > num_pts:
        y_shared = y_shared[:num_pts]
    elif y_shared.shape[0] < num_pts:
        extra = num_pts - y_shared.shape[0]
        z = torch.rand(extra, device=device) * max_z
        x = torch.rand(extra, device=device) * max_x
        y_extra = torch.stack([z, x], dim=-1)
        y_shared = torch.cat([y_shared, y_extra], dim=0)

    return y_shared
```

---

## 6. 训练循环修改方案

### 6.1 在 epoch 开始处计算 `epoch_prob`

原训练循环：

```python
for i in pbar:
    model.train()
    ...
    for batch_data in dataloader['train']:
        ...
```

修改为：

```python
for i in pbar:
    model.train()

    with torch.no_grad():
        epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(
            model=model,
            train_loader=dataloader['train'],
            device=device,
            use_max_mix=False,
        )

    for batch_data in dataloader['train']:
        ...
```

### 6.2 在每个 velocity batch 中采样共享 `y_ran`

原逻辑：

```python
with torch.no_grad():
    y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)
```

替换为：

```python
with torch.no_grad():
    y_shared = sample_shared_y_ran_from_epoch_prob(
        prob=epoch_prob,
        args=args,
        num_pts=900,
        structure_ratio=0.60,
        surface_ratio=0.20,
        uniform_ratio=0.20,
        source_ratio=0.0,
        source_coords=None,
        surface_depth_grids=5,
    )

y_ran = y_shared.unsqueeze(0).expand(
    vel_batch.shape[0], -1, -1
).clone().requires_grad_(True)
```

然后传入：

```python
loss, loss_f, loss_u, loss_r = model.loss(
    vel_batch, y_batch, UU0_batch, labels_batch,
    a, b, c,
    data_norm_coe, pde_norm_coe,
    freq_batch=freq_batch,
    y_ran=y_ran
)
```

### 6.3 如果加入 source-near

需要准备：

```python
source_coords = torch.tensor(
    [
        [z0, x0],
        [z1, x1],
        [z2, x2],
        [z3, x3],
        [z4, x4],
    ],
    dtype=torch.float32,
    device=device
)
```

然后：

```python
with torch.no_grad():
    y_shared = sample_shared_y_ran_from_epoch_prob(
        prob=epoch_prob,
        args=args,
        num_pts=900,
        structure_ratio=0.45,
        surface_ratio=0.20,
        source_ratio=0.20,
        uniform_ratio=0.15,
        source_coords=source_coords,
        surface_depth_grids=5,
        source_r_min_grids=1.5,
        source_r_max_grids=8.0,
    )
```

---

## 7. 推荐加入 Args 配置项

建议在 `Args` 中增加：

```python
# ==========================================
# y_ran epoch-level shared sampling
# ==========================================
use_epoch_shared_y_ran = True

y_ran_num_pts = 900

y_ran_structure_ratio = 0.60
y_ran_surface_ratio = 0.20
y_ran_uniform_ratio = 0.20
y_ran_source_ratio = 0.0

y_ran_surface_depth_grids = 5

y_ran_use_max_mix = False
y_ran_mean_weight = 0.7
y_ran_max_weight = 0.3

# 如果加入 source-near
use_source_near_y_ran = False
y_ran_source_r_min_grids = 1.5
y_ran_source_r_max_grids = 8.0

# 概率图更新频率
# 1 表示每个 epoch 更新一次
# 5 表示每 5 个 epoch 更新一次
# 0 表示只计算一次并缓存
y_ran_prob_update_every = 1
```

如果启用 source-near，则使用：

```python
y_ran_structure_ratio = 0.45
y_ran_surface_ratio = 0.20
y_ran_source_ratio = 0.20
y_ran_uniform_ratio = 0.15
use_source_near_y_ran = True
```

---

## 8. 是否每个 epoch 都重算 epoch_prob？

### 8.1 严格按照“每个 epoch 对所有 vel model 平均采样”

如果严格按当前设想：

```text
每个 epoch 遍历所有训练 velocity model，计算 mean(|∇v|)
```

那么每个 epoch 都调用：

```python
epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(...)
```

### 8.2 计算效率分析

对你的设置：

```text
nvel_train = 1500
batch_size_v = 10
nz = 140
nx = 140
```

每个 epoch 需要计算：

```text
1500 × 140 × 140 ≈ 29,400,000 个网格点的简单差分
```

这个计算相对 FNO + PDE 二阶 autograd 很便宜。

但是，它会额外遍历一次 `dataloader['train']`，可能带来数据加载开销。

### 8.3 推荐实际实现

因为 velocity 数据本身不随 epoch 变化，`epoch_prob` 实际上可以缓存。推荐三种模式：

| 模式 | 说明 | 推荐程度 |
|---|---|---|
| 每 epoch 重算 | 完全符合当前设想，但有重复计算 | 可先实现验证 |
| 每 stage 重算 | 阶段数据变了才重算 | 推荐 |
| 训练开始前重算一次 | 单阶段固定数据最省 | 推荐 |

建议先写成支持参数：

```python
y_ran_prob_update_every = 1
```

实现逻辑：

```python
if epoch_prob is None or (
    args.y_ran_prob_update_every > 0
    and i % args.y_ran_prob_update_every == 0
):
    epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(...)
```

---

## 9. 可视化与诊断建议

建议每隔一定 epoch 保存 `epoch_score` 热力图，确认采样概率是否合理。

### 9.1 保存概率图

```python
np.save(
    os.path.join(args.save_doc, f"epoch_y_ran_score_{i}.npy"),
    epoch_score.detach().cpu().numpy()
)
```

### 9.2 可视化采样点

```python
def plot_y_ran_points(y_shared, args, save_path):
    import matplotlib.pyplot as plt

    y_np = y_shared.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_np[:, 1] / args.dh, y_np[:, 0] / args.dh, s=2)
    plt.gca().invert_yaxis()
    plt.xlim(0, args.nx)
    plt.ylim(args.nz, 0)
    plt.title("Shared y_ran points")
    plt.xlabel("x index")
    plt.ylabel("z index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
```

建议保存：

```text
1. epoch_score 热力图
2. y_shared scatter 图
3. 与某几个 velocity model 的叠加图
```

---

## 10. 与原始 `generate_structure_aware_y_ran` 的关系

### 10.1 不建议直接覆盖原函数

建议保留原函数：

```python
generate_structure_aware_y_ran(...)
```

新增函数：

```python
build_epoch_velocity_gradient_prob(...)
sample_shared_y_ran_from_epoch_prob(...)
```

这样可以方便对比：

```text
原 per-model 采样
batch shared 采样
epoch shared 采样
```

### 10.2 可以在 `Pi_DeepONet` 类中封装

也可以把两个函数作为 `Pi_DeepONet` 的方法：

```python
class Pi_DeepONet(nn.Module):
    ...

    def build_epoch_velocity_gradient_prob(self, train_loader, device, ...):
        ...

    def sample_shared_y_ran_from_epoch_prob(self, prob, num_pts=900, ...):
        ...
```

但从工程上看，放在训练脚本或 `model.utils` 里也可以。

推荐：

```text
如果只用于训练采样：放在 train.py 或 sampling_utils.py
如果希望模型内部统一管理采样：放在 Pi_DeepONet 类中
```

---

## 11. 需要注意的坐标范围

当前很多代码使用：

```python
max_z = args.nz * args.dh
max_x = args.nx * args.dh
```

这适合 cell-style 采样，即：

```text
z_idx ∈ [0, nz-1]
z = z_idx * dh + rand * dh
```

如果你希望采样严格落在节点坐标：

```text
z = z_idx * dh
```

则应使用：

```python
max_z = (args.nz - 1) * args.dh
max_x = (args.nx - 1) * args.dh
```

但当前 `loss()` 中标签提取使用：

```python
z_coord = (y[:, :, 0] / dh).long().clamp(0, nz - 1)
x_coord = (y[:, :, 1] / dh).long().clamp(0, nx - 1)
```

因此采用 cell-style 采样是可以接受的：

```python
z = z_idx * dh + rand * dh
x = x_idx * dh + rand * dh
```

如果你希望所有采样点都严格对应网格点，则改成：

```python
z = z_idx.float() * dh
x = x_idx.float() * dh
```

对 PDE loss 来说，连续 cell 内点更合理；对 data loss 来说，严格网格点更一致。  
由于 `y_ran` 主要服务 PDE loss，推荐保留 cell 内随机扰动。

---

## 12. 与 `model.loss()` 的关系

当前 `model.loss()` 会：

```python
if y_ran is not None:
    y_combined = torch.cat([y, y_ran], dim=1)
else:
    y_combined = y

Delta_U = self.forward(vel, y_combined, UU0)

loss_u = data loss on first n_y points
loss_f = PDE loss on y_combined
```

这意味着：

```text
y_ran 只额外进入 PDE loss；
data loss 仍然只使用 y_batch。
```

这是合理的。

但是要注意：

```text
如果 y_ran 点数增加，PDE 二阶 autograd 成本会增加；
如果 y_ran 分布过于集中，PDE loss 会偏向特定区域；
如果 y_ran 与 y_batch 拼接过大，显存会上升。
```

本计划保持：

```text
num_pts = 900
```

因此计算量基本不变。

---

## 13. 推荐代码整合位置

### 13.1 在训练循环外定义变量

```python
epoch_prob = None
epoch_score = None
```

### 13.2 在每个 epoch 开始处更新

```python
should_update_prob = (
    epoch_prob is None
    or args.y_ran_prob_update_every == 1
    or (
        args.y_ran_prob_update_every > 1
        and i % args.y_ran_prob_update_every == 0
    )
)

if should_update_prob:
    epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(
        model=model,
        train_loader=dataloader['train'],
        device=device,
        use_max_mix=args.y_ran_use_max_mix,
        mean_weight=args.y_ran_mean_weight,
        max_weight=args.y_ran_max_weight,
    )
```

### 13.3 在每个 vel_batch 内采样

```python
with torch.no_grad():
    y_shared = sample_shared_y_ran_from_epoch_prob(
        prob=epoch_prob,
        args=args,
        num_pts=args.y_ran_num_pts,
        structure_ratio=args.y_ran_structure_ratio,
        surface_ratio=args.y_ran_surface_ratio,
        uniform_ratio=args.y_ran_uniform_ratio,
        source_ratio=args.y_ran_source_ratio,
        source_coords=source_coords if args.use_source_near_y_ran else None,
        surface_depth_grids=args.y_ran_surface_depth_grids,
        source_r_min_grids=args.y_ran_source_r_min_grids,
        source_r_max_grids=args.y_ran_source_r_max_grids,
    )

y_ran = y_shared.unsqueeze(0).expand(
    vel_batch.shape[0], -1, -1
).clone().requires_grad_(True)
```

---

## 14. 推荐实验设计

### 14.1 Baseline

```text
原始 y_ran：
50% per-batch/per-model structure
50% surface
```

或者你当前实际可运行版本。

### 14.2 Experiment A：epoch mean structure

```text
60% epoch mean structure
20% surface
20% uniform
surface_depth = 5dh
```

### 14.3 Experiment B：epoch mean+max structure

```text
60% epoch mean+max structure
20% surface
20% uniform
score = 0.7 mean + 0.3 max
surface_depth = 5dh
```

### 14.4 Experiment C：加入 source-near

```text
45% epoch mean structure
20% surface
20% source-near
15% uniform
surface_depth = 5dh
source radius = [1.5dh, 8dh]
```

### 14.5 Experiment D：mean+max + source-near

```text
45% epoch mean+max structure
20% surface
20% source-near
15% uniform
```

### 14.6 推荐优先级

```text
1. Experiment A
2. Experiment B
3. Experiment C
4. Experiment D
```

如果震源附近误差明显，则 C/D 可以提前。

---

## 15. 评价指标

建议比较：

```text
1. 训练 data loss
2. 训练 PDE loss
3. 验证 relative L2
4. envelope relative L2
5. complex correlation
6. 高频波场可视化
7. 震源附近误差
8. 速度界面附近误差
9. 表层误差
```

如果只是 PDE loss 降了，但 data / 可视化变差，说明采样分布可能过度偏向 PDE 局部区域。

---

## 16. 计算效率预估

### 16.1 新增开销

每个 epoch 增加一次全训练集速度梯度统计：

```text
O(nvel_train × nz × nx)
```

你的配置约为：

```text
1500 × 140 × 140 ≈ 2.94e7 简单差分操作
```

这比 PDE 二阶 autograd 便宜很多。

### 16.2 每个 batch 的采样开销

从 `epoch_prob` 采样：

```text
torch.multinomial over 19600 elements
sample 400–600 points
```

非常便宜。

### 16.3 真正的瓶颈

仍然是：

```text
y_combined 上的 forward
PDE residual 的一阶/二阶 autograd
loss.backward()
```

只要 `y_ran_num_pts=900` 不增加，总训练计算量基本不变。

---

## 17. 关键注意事项

### 17.1 采样过程必须放在 `torch.no_grad()`

```python
with torch.no_grad():
    y_shared = sample_shared_y_ran_from_epoch_prob(...)
```

否则会产生无意义计算图。

### 17.2 expand 后要 clone

```python
y_ran = y_shared.unsqueeze(0).expand(B, -1, -1).clone().requires_grad_(True)
```

不要直接对 expand view 做 requires_grad 后复杂操作。

### 17.3 不要在 train_y 内层循环重复计算 epoch_prob

`epoch_prob` 应该在 epoch 开始计算一次。

### 17.4 如果使用 Halton 内层循环，y_ran 应该每个 vel_batch 生成一次

推荐：

```python
with torch.no_grad():
    y_shared = ...
y_ran = ...

for batch in dataloader['train_y']:
    loss(..., y_ran=y_ran)
```

不要在每个 `train_y` batch 内重新生成。

### 17.5 保存概率图做检查

必须检查 `epoch_score` 是否合理。否则采样可能集中在不希望的位置。

---

## 18. 最小可行修改版本

如果只想做最小修改，推荐：

### Step 1：新增两个函数

```python
build_epoch_velocity_gradient_prob(...)
sample_shared_y_ran_from_epoch_prob(...)
```

### Step 2：在 epoch 开始处计算

```python
epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(...)
```

### Step 3：替换原来的 y_ran 生成

```python
with torch.no_grad():
    y_shared = sample_shared_y_ran_from_epoch_prob(
        prob=epoch_prob,
        args=args,
        num_pts=900,
        structure_ratio=0.60,
        surface_ratio=0.20,
        uniform_ratio=0.20,
        source_ratio=0.0,
        source_coords=None,
        surface_depth_grids=5,
    )

y_ran = y_shared.unsqueeze(0).expand(
    vel_batch.shape[0], -1, -1
).clone().requires_grad_(True)
```

### Step 4：保持 loss 逻辑不变

```python
loss(..., y_ran=y_ran)
```

---

## 19. 推荐最终第一版配置

```python
use_epoch_shared_y_ran = True

y_ran_num_pts = 900

y_ran_structure_ratio = 0.60
y_ran_surface_ratio = 0.20
y_ran_uniform_ratio = 0.20
y_ran_source_ratio = 0.0

y_ran_surface_depth_grids = 5

y_ran_use_max_mix = False
y_ran_mean_weight = 0.7
y_ran_max_weight = 0.3

y_ran_prob_update_every = 1
```

如果验证发现平均结构图过于平滑，再改：

```python
y_ran_use_max_mix = True
```

如果加入 source-near：

```python
y_ran_structure_ratio = 0.45
y_ran_surface_ratio = 0.20
y_ran_source_ratio = 0.20
y_ran_uniform_ratio = 0.15
use_source_near_y_ran = True
```

---

## 20. 总结

本修改方案将 `y_ran` 从当前 batch/per-model 结构采样，调整为：

```text
每个 epoch 基于所有训练 velocity model 的平均速度梯度图构造全局结构概率分布；
每个 velocity batch 从该概率分布采样一组共享 y_ran；
batch 内所有速度模型共享这组 y_ran；
y_ran 继续只作为 PDE residual 的额外采样点。
```

核心收益：

```text
1. 满足当前显存限制；
2. 避免 y_ran 偏向 batch 内某一个 velocity；
3. 比 per-model adaptive sampling 更稳定、更省；
4. 比完全 uniform / surface 采样更关注结构区域；
5. 方便加入 surface、uniform、source-near 混合采样；
6. 不改变 loss 主体逻辑，工程风险较低。
```

最推荐第一版：

```text
60% epoch mean structure
20% surface, depth = 5dh
20% uniform
num_pts = 900
```

后续再逐步尝试：

```text
mean+max structure
source-near points
不同 surface_depth
不同 y_ran 比例
```
