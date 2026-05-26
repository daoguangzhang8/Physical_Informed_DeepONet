# Branch2 CNN Replacement Design

## Problem

Branch2 currently uses FNO2d to extract features from the background wavefield UU0. FNO operates in the Fourier domain and tends to capture global low-frequency patterns, potentially missing local spatial features such as wavefront shapes near sources and interference fringes.

## Decision

Add CNN-based alternatives (ResNet-style and simple Conv stacking) for branch2, selectable via config parameter. Keep FNO as baseline option.

## Architecture

### ResNetBranch2d

```
Input: [B, 2, Z, X]
  Stem:    Conv2d(2, 64, 3x3, pad=1) -> BN -> ReLU
  Stage1:  2x ResBlock(64)
  Expand:  Conv2d(64, 128, 3x3, pad=1) -> BN -> ReLU
  Stage2:  2x ResBlock(128)
  Output:  Conv2d(128, 256, 3x3, pad=1)
Output: [B, 256, Z, X]
```

ResBlock: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + Identity shortcut. All convolutions use padding=1 to preserve spatial resolution.

### ConvBranch2d

```
Input: [B, 2, Z, X]
  Conv2d(2, 32, 3x3, pad=1)   -> BN -> ReLU
  Conv2d(32, 64, 3x3, pad=1)  -> BN -> ReLU
  Conv2d(64, 128, 3x3, pad=1) -> BN -> ReLU
  Conv2d(128, 256, 3x3, pad=1)-> BN -> ReLU
  Conv2d(256, 256, 3x3, pad=1)
Output: [B, 256, Z, X]
```

### FNO2d (unchanged)

Existing implementation, kept as baseline.

## Config Parameter

```python
branch2_type = 'resnet'  # 'fno' | 'resnet' | 'conv'
```

Default value: `'fno'` to maintain backward compatibility.

## File Changes

| File | Change |
|------|--------|
| `model/net_module.py` | Add `ResNetBranch2d` and `ConvBranch2d` classes |
| `model/PI_DeepOnet.py` | Branch2 instantiation logic in `__init__` |
| `config.py` | Add `branch2_type` parameter |

## Unchanged Components

- `branch1` (FNO for velocity field)
- `forward()` method - branch2 call signature unchanged
- `channel_attention2`, `combinedlayer2`, `attengate` - input shape preserved
- `loss()`, `_compute_pde_residual()` - no branch2 involvement
- `train.py`, `dataloader.py`, all other files

## Backward Compatibility

When `branch2_type='fno'` (or omitted), behavior is identical to current code. No breaking changes.
