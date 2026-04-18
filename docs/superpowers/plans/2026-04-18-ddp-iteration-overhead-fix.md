# DDP Iteration Overhead Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate two sources of per-iteration overhead in DDP training so 2-GPU training achieves near-theoretical 2x speedup.

**Architecture:** Remove 3 unused parameters from model classes (`Pi_DeepONet`, `FNO`) to allow `find_unused_parameters=False`. Move coordinate DataLoader materialization outside the velocity loop to eliminate redundant re-creation.

**Tech Stack:** PyTorch DDP (nccl), torch.distributed

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `model/PI_DeepOnet.py` | Modify | Delete `block_feature_encoder`, `log_var_data`, `log_var_pde` |
| `model/FNO.py` | Modify | Delete `log_var_data`, `log_var_pde` |
| `model/utils.py` | Modify | Set `find_unused_parameters=False` |
| `model/train_distributed.py` | Modify | Move `coord_batches` outside velocity loop |

---

### Task 1: Remove unused parameters from Pi_DeepONet

**Files:**
- Modify: `model/PI_DeepOnet.py` (lines 46, 58-59)

- [ ] **Step 1: Delete `block_feature_encoder` definition**

In `model/PI_DeepOnet.py`, delete line 46:

```python
# DELETE this line:
        self.block_feature_encoder = BlockFeatureEncoder(self.feat_dim, self.feat_dim, grid_size=20)  # 未使用，已注释
```

The surrounding context should change from:
```python
        self.attengate = AttenGate(use_softmax=True)
        
        self.block_feature_encoder = BlockFeatureEncoder(self.feat_dim, self.feat_dim, grid_size=20)  # 未使用，已注释
        self.smooth_feature_encoder = SmoothBlockEncoder(self.feat_dim, self.feat_dim, grid_size=20)
```
To:
```python
        self.attengate = AttenGate(use_softmax=True)
        
        self.smooth_feature_encoder = SmoothBlockEncoder(self.feat_dim, self.feat_dim, grid_size=20)
```

- [ ] **Step 2: Delete `log_var_data` and `log_var_pde` parameters**

In `model/PI_DeepOnet.py`, delete lines 57-59:

```python
# DELETE these 3 lines:
        # 动态损失权重参数
        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_pde = nn.Parameter(torch.zeros(1))
```

The surrounding context should change from:
```python
        # self.loss_function_point = nn.MSELoss(reduction='none')  # 未使用，已注释
        
        # 动态损失权重参数
        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_pde = nn.Parameter(torch.zeros(1))

        self._init_weights()
```
To:
```python
        # self.loss_function_point = nn.MSELoss(reduction='none')  # 未使用，已注释

        self._init_weights()
```

- [ ] **Step 3: Verify import succeeds**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.PI_DeepOnet import Pi_DeepONet; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 4: Commit**

```bash
git add model/PI_DeepOnet.py
git commit -m "perf: remove unused parameters from Pi_DeepONet (block_feature_encoder, log_var_data, log_var_pde)"
```

---

### Task 2: Remove unused parameters from FNO

**Files:**
- Modify: `model/FNO.py` (lines 28-29)

- [ ] **Step 1: Delete `log_var_data` and `log_var_pde` parameters**

In `model/FNO.py`, delete lines 28-29:

```python
# DELETE these 2 lines:
        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_pde = nn.Parameter(torch.zeros(1))
```

The surrounding context should change from:
```python
        self.pde_imag_k = 0.

        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_pde = nn.Parameter(torch.zeros(1))
        
        self.FNO = nn.Sequential(
```
To:
```python
        self.pde_imag_k = 0.

        self.FNO = nn.Sequential(
```

- [ ] **Step 2: Verify import succeeds**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.FNO import FNO; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add model/FNO.py
git commit -m "perf: remove unused parameters from FNO (log_var_data, log_var_pde)"
```

---

### Task 3: Disable find_unused_parameters in DDP wrapper

**Files:**
- Modify: `model/utils.py` (line 283)

- [ ] **Step 1: Change `find_unused_parameters` to `False`**

In `model/utils.py`, change line 283 from:

```python
    model = DDP.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
```

To:

```python
    model = DDP.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
```

- [ ] **Step 2: Verify import succeeds**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.utils import wrap_model_for_distributed; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add model/utils.py
git commit -m "perf: set find_unused_parameters=False now that all model params are used in forward"
```

---

### Task 4: Move coord_batches materialization outside velocity loop

**Files:**
- Modify: `model/train_distributed.py` (lines 171-198)

- [ ] **Step 1: Move coord_batches outside the velocity loop**

In `model/train_distributed.py`, replace lines 171-198 (from `dataloader['train'].sampler.set_epoch(i)` through the start of the inner loop) with:

From (current):
```python
        # 设置 epoch 以确保每个 epoch shuffle 不同
        dataloader['train'].sampler.set_epoch(i)

        # 遍历训练数据
        for batch_data in dataloader['train']:
            if has_freq:
                vel_batch, UU0_batch, labels_batch, freq_batch = batch_data
                freq_batch = freq_batch.to(device)
            else:
                vel_batch, UU0_batch, labels_batch = batch_data
                freq_batch = None
            vel_batch, UU0_batch = vel_batch.to(device), UU0_batch.to(device)

            if args.use_fno_as_label:
                with torch.no_grad():
                    labels_batch = fno(vel_batch, UU0_batch).to(device)
            else:
                labels_batch = labels_batch.to(device)

            # 每个 velocity batch 只生成一次自适应采样点
            with torch.no_grad():
                y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)

            # 预收集 coordinate batches 以确定总数
            coord_batches = list(dataloader['train_y'])
            n_coord = len(coord_batches)
            

            for idx, batch in enumerate(coord_batches):
```

To:
```python
        # 设置 epoch 以确保每个 epoch shuffle 不同
        dataloader['train'].sampler.set_epoch(i)

        # 预收集 coordinate batches（每 epoch 一次，避免在 velocity 循环内重复物化）
        coord_batches = list(dataloader['train_y'])
        n_coord = len(coord_batches)

        # 遍历训练数据
        for batch_data in dataloader['train']:
            if has_freq:
                vel_batch, UU0_batch, labels_batch, freq_batch = batch_data
                freq_batch = freq_batch.to(device)
            else:
                vel_batch, UU0_batch, labels_batch = batch_data
                freq_batch = None
            vel_batch, UU0_batch = vel_batch.to(device), UU0_batch.to(device)

            if args.use_fno_as_label:
                with torch.no_grad():
                    labels_batch = fno(vel_batch, UU0_batch).to(device)
            else:
                labels_batch = labels_batch.to(device)

            # 每个 velocity batch 只生成一次自适应采样点
            with torch.no_grad():
                y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)

            for idx, batch in enumerate(coord_batches):
```

- [ ] **Step 2: Verify import succeeds**

Run: `cd /home/zhangdaoguang/Code/Physical_Informed_DeepONet && python -c "from model.train_distributed import train_distributed; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add model/train_distributed.py
git commit -m "perf: move coord_batches materialization outside velocity loop"
```
