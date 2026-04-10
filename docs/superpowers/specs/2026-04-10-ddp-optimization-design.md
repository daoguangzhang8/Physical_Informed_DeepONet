# DDP Distributed Training Optimization Design

**Date:** 2026-04-10
**Scope:** Fix DDP gradient synchronization bugs in single-machine multi-GPU training
**Target:** 2x GPU (single machine)

## Problem Statement

Single-machine multi-GPU DDP training is slower than single-GPU. Root cause analysis revealed three critical bugs that prevent DDP from functioning correctly:

1. **No gradient synchronization**: `model.module.loss()` bypasses the DDP wrapper, so all-reduce never fires. Each GPU trains independently.
2. **Weight initialization race condition**: Only rank 0 calls `_init_weights()`, other ranks start with random weights and are never synchronized.
3. **Inefficient gradient sync pattern**: The `no_sync` + `accumulation_steps` logic is moot because DDP hooks are never registered (consequence of bug #1).

## Constraints

- **Inner loop must remain**: Memory constraints require iterating coordinate batches sequentially within each velocity batch.
- **2 GPUs only**: Optimization targets single-machine 2-GPU setup.
- **Validation unchanged**: Only rank 0 runs validation/plotting/saving.
- **LR scaling and find_unused_parameters unchanged**: These are correct or acceptable as-is.
- **No mmap changes**: Data loading strategy remains as-is for now.

## Design

### Section 1: Fix DDP Forward Pass

**File:** `model/PI_DeepOnet.py`

Split `loss()` into two methods:

- `forward(vel, y, UU0)` — unchanged, pure neural network forward pass
- `compute_loss(outputs, vel, y_data, y_ran, labels, y_grid, ...)` — new method containing data loss + PDE residual computation, no forward call

**File:** `model/train_distributed.py`

Change training loop from:
```python
loss, loss_f, loss_u, loss_r = model.module.loss(vel_batch, y_batch, UU0_batch, ...)
loss.backward()
```

To:
```python
outputs = model(vel, y_combined, UU0)                    # Through DDP wrapper
loss, loss_f, loss_u, loss_r = model.module.compute_loss(outputs, ...)
loss.backward()                                           # DDP all-reduce triggers
```

**Rationale:** DDP registers gradient synchronization hooks during `forward()`. Calling `model.module.loss()` bypasses this registration, making all-reduce impossible.

**Autograd compatibility:** `_compute_pde_residual()` uses `torch.autograd.grad` for second derivatives. The computation graph connects `outputs` to `model()` forward, so gradients flow correctly through the full path: loss → PDE residual → outputs → model parameters.

### Section 2: Gradient Sync Strategy (no_sync + Inner Accumulation)

**File:** `model/train_distributed.py`

Restructure the training loop to synchronize once per velocity batch instead of (theoretically) once per coordinate batch:

```python
for vel_batch in dataloader['train']:                    # DistributedSampler shards velocities
    y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)

    coord_batches = list(dataloader['train_y'])
    n_coord = len(coord_batches)

    for idx, coord_batch in enumerate(coord_batches):
        y_batch = coord_batch[0].to(device)
        y_combined = torch.cat([y_batch, y_ran], dim=1)

        outputs = model(vel, y_combined, UU0)            # DDP forward
        loss, lf, lu, lr = model.module.compute_loss(...)
        loss = loss / n_coord                             # Average across coord batches

        if idx < n_coord - 1:
            with model.no_sync():
                loss.backward()                           # No all-reduce, just accumulate
        else:
            loss.backward()                               # All-reduce fires here

    optimizer.step()
    optimizer.zero_grad()
```

**Key changes:**
- `loss / n_coord` compensates for accumulating gradients across n_coord coordinate batches (equivalent to averaging)
- `optimizer.step()` and `zero_grad()` moved to outer loop end (once per velocity batch)
- Original `accumulation_steps` logic removed (no longer needed)
- Communication reduced from `num_vel_batches * num_coord_batches` to `num_vel_batches`

### Section 3: Weight Initialization Broadcast

**File:** `model/train_distributed.py`

After rank 0 initializes weights, broadcast all parameters to other ranks before wrapping with DDP:

```python
model = Pi_DeepONet(args).to(device)

if is_main_process(rank):
    model._init_weights()                                # Only rank 0 initializes

# Broadcast all parameters from rank 0 to other ranks
for param in model.parameters():
    dist.broadcast(param.data, src=0)

model = wrap_model_for_distributed(model, rank)          # DDP wrapper after broadcast
```

**Rationale:** DDP snapshots model state at wrapper construction time. Broadcasting before wrapping ensures all ranks start with identical parameters.

## Files Modified

| File | Changes |
|------|---------|
| `model/PI_DeepOnet.py` | Add `compute_loss()` method extracted from `loss()` |
| `model/train_distributed.py` | Fix forward call, gradient sync pattern, weight init broadcast |
| `model/train.py` | No changes (single-GPU path unaffected) |

## Expected Outcome

- DDP gradient synchronization works correctly for the first time
- 2-GPU training should achieve ~1.7-1.9x speedup over single-GPU
- Training convergence behavior should match single-GPU (same loss landscape, proper gradient averaging)
