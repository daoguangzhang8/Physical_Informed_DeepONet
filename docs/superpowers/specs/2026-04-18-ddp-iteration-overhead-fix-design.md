# DDP Per-Iteration Overhead Fix — Design

**Date:** 2026-04-18
**Scope:** Eliminate two sources of per-iteration overhead in DDP training that cause 60% slowdown per iteration vs single GPU, reducing the expected 2x speedup to only 1.25x.
**Target:** 2x GPU (single machine), `Physical_Informed_DeepONet` project

## Problem Statement

DDP training with 2 GPUs achieves only ~1.25x speedup (1000s → 800s per epoch) instead of the theoretical ~2x. Root cause: each DDP iteration takes ~142ms vs ~89ms on single GPU (+60%). Two fixable sources identified:

1. **`find_unused_parameters=True` overhead**: The model defines 3 parameters (`block_feature_encoder`, `log_var_data`, `log_var_pde`) that are never used in `forward()` or `compute_loss()`. DDP with `find_unused_parameters=True` must traverse the entire autograd graph on every `backward()` call to identify unused parameters. With 4 `torch.autograd.grad(create_graph=True)` calls building complex second-order derivative graphs, this traversal is extremely expensive — fired 5,628 times per epoch.

2. **`coord_batches = list(dataloader['train_y'])` inside velocity loop**: Materializes the entire coordinate DataLoader into a Python list on every velocity batch iteration (~469 times per epoch per GPU), creating redundant object allocations.

## Constraints

- **No backward compatibility needed**: Old checkpoint weights do not need to be loadable after this change.
- **Single-GPU train.py path unchanged**: Only `train_distributed.py` and model definitions are modified.
- **Validation/plotting/saving paths unaffected**: Only the training hot loop is changed.

## Design

### Section 1: Remove Unused Model Parameters

**Files modified:** `model/PI_DeepOnet.py`, `model/FNO.py`, `model/utils.py`

**Delete from `model/PI_DeepOnet.py`:**
- Line 46: `self.block_feature_encoder = BlockFeatureEncoder(self.feat_dim, self.feat_dim, grid_size=20)`
- Line 58: `self.log_var_data = nn.Parameter(torch.zeros(1))`
- Line 59: `self.log_var_pde = nn.Parameter(torch.zeros(1))`

**Delete from `model/FNO.py`:**
- Line 28: `self.log_var_data = nn.Parameter(torch.zeros(1))`
- Line 29: `self.log_var_pde = nn.Parameter(torch.zeros(1))`

**Modify `model/utils.py` line 283:**
```python
# Before:
model = DDP.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
# After:
model = DDP.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
```

**Rationale:** After removing all unused parameters, every model parameter participates in the forward pass. DDP can safely use `find_unused_parameters=False`, eliminating the per-backward graph traversal overhead. This also slightly reduces model parameter count and memory usage.

**Impact:** Expected to eliminate the ~60% per-iteration overhead caused by DDP graph traversal, bringing per-iteration time from ~142ms back to ~89ms. Epoch time should drop from ~800s to ~500s, achieving near-theoretical 2x speedup.

### Section 2: Move coord_batches Materialization Outside Velocity Loop

**File modified:** `model/train_distributed.py`

**Current code (inside velocity loop, ~line 193-194):**
```python
for batch_data in dataloader['train']:
    ...
    coord_batches = list(dataloader['train_y'])  # called ~469 times per epoch per GPU
    n_coord = len(coord_batches)
    for idx, batch in enumerate(coord_batches):
```

**New code (move outside velocity loop, after `dataloader['train'].sampler.set_epoch(i)`):**
```python
coord_batches = list(dataloader['train_y'])
n_coord = len(coord_batches)

for batch_data in dataloader['train']:
    ...
    for idx, batch in enumerate(coord_batches):
```

**Rationale:** `dataloader['train_y']` is a `DataLoader(TensorDataset(y_train), batch_size=800, shuffle=True)`. Each `list()` call iterates through 12 batches. While each call is fast (~1ms), calling it 469 times per epoch per GPU adds up to ~0.5s of pure overhead. Moving it outside the loop reduces this to one call per epoch.

**Trade-off:** All velocity batches within one epoch now share the same coordinate ordering. This is acceptable because:
- The DataLoader shuffles on every new iteration, so each epoch gets different ordering
- The coordinate batches serve as spatial sampling points — their ordering doesn't affect training semantics
- Single-GPU `train.py` already iterates the same `train_y` DataLoader in the same pattern

**Impact:** Minor (~0.5s/epoch), but eliminates unnecessary object churn in the hot loop.

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Per-iteration time (DDP) | ~142ms | ~89ms |
| Epoch time (DDP 2 GPU) | ~800s | ~500s |
| Speedup vs single GPU | 1.25x | ~2.0x |
| `find_unused_parameters` | True | False |

## Files Modified

| File | Change |
|------|--------|
| `model/PI_DeepOnet.py` | Delete 3 unused parameter definitions |
| `model/FNO.py` | Delete 2 unused parameter definitions |
| `model/utils.py` | Set `find_unused_parameters=False` |
| `model/train_distributed.py` | Move `coord_batches` materialization outside velocity loop |
| `model/train.py` | No changes (single-GPU path unaffected) |
