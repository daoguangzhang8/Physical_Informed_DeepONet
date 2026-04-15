# Sobol 采样消除内层循环设计

**日期**: 2026-04-15
**状态**: 待实施
**基线 commit**: `1d55d23`

## 目标

将训练中的双层嵌套循环（外层 velocity batch + 内层坐标 batch）改为单层循环，用 Sobol 准随机采样替代 Halton 固定坐标集，每 epoch 仅做一次前向+反传+参数更新。

## 背景

### 当前训练循环结构

```
for vel_batch, UU0_batch, labels_batch in train:        # 外层 (~5 样本)
    y_ran = generate_structure_aware_y_ran(vel, 900)    # 结构感知PDE点
    for y_batch in train_y:                              # 内层 (~12-13 次)
        y_combined = cat(y_batch, y_ran)                 # ~1700 点
        loss = model.loss(vel, y_combined, UU0, labels)
        loss.backward()                                  # 梯度累加
        if step_counter % accumulation_steps == 0:
            optimizer.step()
```

问题：内层循环导致每个 velocity batch 需要 ~12 次前向传播，训练速度慢。

### Sobol 覆盖率实验结论

在 140×140=19600 网格上，每 epoch 采样 800 个 Sobol 点：

| 累积 epoch | 累积采样量 | Sobol 覆盖率 |
|-----------|-----------|-------------|
| 15        | 12000     | >50%        |
| 50        | 40000     | >98%        |
| 150       | 120000    | 100%        |

训练共 5000+ epoch，Sobol 覆盖远在训练结束前饱和。

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 梯度更新策略 | 每 epoch 直接更新 | 消除内层循环后不再需要梯度累加 |
| PDE 采样点 | 保留 900 个结构感知点 | 物理约束在结构敏感区域（速度梯度边界、自由表面）仍然必要 |
| 验证集 | 同步改 Sobol，点数由 `valid_sobol_points` 控制 | 与训练采样方式一致 |
| Sobol 引擎 | 持久化，跨 epoch 序列延续 | 累积覆盖最优，无额外开销 |
| 旧代码 | 完整保留 | 新逻辑处于测试阶段 |
| 改动方式 | 直接修改 `train.py` | 用户明确要求 |

## 改造后训练循环结构

```
sobol_engine = SobolEngine(dim=2, scramble=True)          # 持久化

for vel_batch, UU0_batch, labels_batch in train:           # 唯一循环
    y_sobol = sobol_engine.draw(800)                       # Sobol 连续序列
    y_sobol = y_sobol * [nz*dh, nx*dh]                     # 映射到物理坐标
    y_sobol.requires_grad_(True)

    y_ran = model.generate_structure_aware_y_ran(vel, 900)  # 结构感知PDE点
    y_combined = cat([y_sobol, y_ran], dim=1)               # ~1700 点

    loss = model.loss(vel, y_combined, UU0, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 文件变更范围

### `config.py` — 追加参数

```python
# ==========================================
# 10. Sobol 采样配置 (新增)
# ==========================================
sampling_strategy = 'sobol'               # 'original': 双层循环+Halton | 'sobol': 单层循环
sobol_points_per_epoch = 800              # 每 epoch Sobol 数据采样点数
valid_sobol_points = 800                  # 验证集 Sobol 采样点数
```

不动任何已有参数。

### `model/train.py` — 核心改造

改造点：
1. 训练循环开头创建 `SobolEngine`（持久化）和验证用 `SobolEngine`
2. 主循环中根据 `sampling_strategy` 走不同分支：
   - `'sobol'`：单层循环，800 Sobol + 900 结构感知 → 一次前向 → 直接更新
   - `'original'`：保持原有双层循环逻辑不变
3. 验证环节同理，`'sobol'` 模式用 `valid_sobol_engine.draw(valid_sobol_points)` 替代 `valid_y` DataLoader
4. 移除 `'sobol'` 模式下的 `accumulation_steps` 逻辑和 `step_counter`
5. 可视化/保存逻辑不变（仍用全网格 `pred`/`test` DataLoader）

### `main2.py` — 无需修改

`main2.py` 只调用 `train(args)`，内部由 `args.sampling_strategy` 决定走哪条路径。

### 其他文件 — 不动

`model/PI_DeepOnet.py`、`model/dataloader.py`、`model/net_module.py` 等均不修改。

## Sobol 坐标映射细节

`SobolEngine.draw(N)` 输出 [0, 1)^2 的均匀点：

```python
pts = sobol_engine.draw(sobol_points_per_epoch)        # [N, 2], 范围 [0, 1)
pts_physical = pts * torch.tensor([nz * dh, nx * dh])  # 映射到物理坐标 [0, nz*dh) × [0, nx*dh)
pts_physical.requires_grad_(True)                       # PDE loss 需要对 y 求导
```

注意 `nz`/`nx` 是 PML 裁剪后的实际尺寸，`dh` 是网格间距。

## 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 单次前向 1700 点显存不够 | OOM | 可调低 `sobol_points_per_epoch` 或结构感知点数 |
| Sobol 采样前期覆盖率不足 | Data Loss 监督信号稀疏 | 15 epoch 后 >50%，对 5000 epoch 训练影响可忽略 |
| SobolEngine 跨 epoch 状态 | 引擎内部计数器持续增长 | `SobolEngine` 支持任意长度序列，无上限问题 |
| 旧逻辑兼容性 | 切回 original 模式需要验证 | 不动任何旧代码，仅新增分支 |
