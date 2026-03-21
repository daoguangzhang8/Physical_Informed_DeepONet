

from Labconfig import *
from torch.optim.lr_scheduler import _LRScheduler

def load_or_save_checkpoint(vit, optimizer, net_opt, DEVICE):
    """
    Load a checkpoint if it exists, otherwise save the initial checkpoint.

    Args:
        vit (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        net_opt (str): The path to the checkpoint file.
        isa (int): The current iteration number.
        DEVICE (torch.device): The device (CPU or GPU) to map the loaded checkpoint to.
    """
    # Load checkpoint if exists
    if os.path.exists(net_opt):
        chk = torch.load(net_opt, map_location=DEVICE)
        print('chk - keys -', chk.keys())
        vit.load_state_dict(chk['vit'])
        optimizer.load_state_dict(chk['opt'])

        # Move the optimizer's state tensors to the specified device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        #isa = chk['iteration'] + 1  # Resume from next iteration
        print(f"Resuming training from iteration isa")
    else:

        #seed_everything(42)
        # Save initial checkpoint
        torch.save({
            'vit': vit.state_dict(),
            'opt': optimizer.state_dict(),
            #'iteration': isa
        }, net_opt)
        print(f"Starting new training from iteration ")
class WarmupScheduler(_LRScheduler):
    """
    学习率热身调度器：先在warmup_epochs内从初始学习率（warmup_start_lr）线性/余弦增长到基础学习率（base_lr），
    之后可衔接其他调度器（如StepLR、CosineAnnealingLR等）。
    
    Args:
        optimizer: 优化器
        warmup_epochs: 热身轮次（如5、10）
        base_lr: 热身结束后的目标学习率（即优化器初始lr）
        warmup_start_lr: 热身起始学习率（通常为base_lr的1/10或1/100，如0.0001）
        warmup_strategy: 热身策略，可选"linear"（线性增长）或"cosine"（余弦增长）
        after_scheduler: 热身结束后使用的调度器（如StepLR，可选）
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, warmup_start_lr=0.0, 
                 warmup_strategy="linear", after_scheduler=None):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_strategy = warmup_strategy
        self.after_scheduler = after_scheduler  # 热身结束后的调度器
        self.current_epoch = 0  # 记录当前轮次
        super().__init__(optimizer)
        
    def get_lr(self):
        # 热身阶段：调整学习率
        if self.current_epoch < self.warmup_epochs:
            if self.warmup_strategy == "linear":
                # 线性增长：lr = start_lr + (base_lr - start_lr) * (current_epoch / warmup_epochs)
                progress = self.current_epoch / self.warmup_epochs
                lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
            elif self.warmup_strategy == "cosine":
                # 余弦增长：lr = start_lr + (base_lr - start_lr) * (1 - cos(pi * current_epoch / (2*warmup_epochs))) / 2
                progress = self.current_epoch / self.warmup_epochs
                lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                     (1 - torch.cos(torch.tensor(torch.pi * progress / 2))) / 2
            else:
                raise ValueError(f"不支持的热身策略：{self.warmup_strategy}，可选'linear'或'cosine'")
            return [lr for _ in self.base_lrs]
        # 热身结束后：使用后续调度器
        else:
            if self.after_scheduler is not None:
                # 后续调度器的 epoch 从 0 开始计数（相对于热身结束）
                self.after_scheduler.step(self.current_epoch - self.warmup_epochs)
                return self.after_scheduler.get_lr()
            else:
                # 若无后续调度器，保持 base_lr
                return [self.base_lr for _ in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        self.current_epoch = epoch
        # 调用父类方法更新学习率
        super().step()

def count_parameters(model):
    """
    计算 PyTorch 模型的总参数数量
    Args:
        model: PyTorch 模型（nn.Module 子类）
    Returns:
        total_params: 模型总参数数量（int）
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# 分布式训练初始化
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def calculate_regression_metrics(pred, true):
    """
    计算回归任务核心指标，包括 Relative L2 Loss
    Args:
        pred: 预测值数组 (nz, nx)
    	true: 真实值数组 (nz, nx)
    Returns:
        metrics: 字典，包含 mse, mae, r2, relative_l2
    """
    assert pred.shape == true.shape, "预测值和真实值维度必须一致"
    
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    # 1. 基础指标
    mse = np.mean((pred_flat - true_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - true_flat))
    
    # 2. R² (决定系数)
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # 3. Relative L2 Loss (相对 L2 误差)
    # 公式: ||pred - true||2 / ||true||2
    # np.linalg.norm 默认计算 L2 范数
    norm_diff = np.linalg.norm(true_flat - pred_flat)
    norm_true = np.linalg.norm(true_flat)
    
    # 避免分母为 0
    relative_l2 = (norm_diff / norm_true) if norm_true != 0 else 0.0
    
    return {
        "mse": relative_l2,
        "mae": mae,
        "r2": r2
    }
def Halton_Sample(array_shape, num_samples):
    sampler = qmc.Halton(d=2, scramble=True)
    samples = sampler.random(num_samples)
    rows = (samples[:, 0] * array_shape[0]).astype(int)
    cols = (samples[:, 1] * array_shape[1]).astype(int)
    rows = np.clip(rows, 0, array_shape[0]-1)
    cols = np.clip(cols, 0, array_shape[1]-1)
    return list(zip(rows, cols))


def generate_random_points(batch_size, n_pts, range_max=72.0):
    # 生成 [B, n_pts, 2] 的随机坐标，范围在 [0, 1] 之间
    y_random = torch.rand(batch_size, n_pts, 2)
    
    # 缩放到 [0, 72] 范围
    y_random = y_random * range_max
    
    # 记得开启梯度跟踪，否则无法计算 PDE Loss 中的导数
    y_random.requires_grad_(True)
    
    return y_random # 输出形状 [B, n_pts, 2]
    
def get_local_physical_features(vel, y, eps=1e-3):
    # 关键点：在采样前，将坐标 y 从计算图中分离 (detach)
    # 这告诉 PyTorch：我只需要在坐标 y 处的值，不需要计算通过 grid_sample 对 y 的梯度
    y_detached = y.detach() 
    
    grid_center = y_detached.unsqueeze(2) 
    
    # 执行采样（现在不需要担心二阶导数问题了）
    v_center = F.grid_sample(vel, grid_center, align_corners=True).squeeze(-1)
    b1 = vel.shape[0]        # Batch size (B_v)
    b_pts = y.shape[1]
    # 同样地采样周围点...
    grid_right = grid_center + torch.tensor([eps, 0], device=y.device)
    grid_up    = grid_center + torch.tensor([0, eps], device=y.device)
    
    v_right = F.grid_sample(vel, grid_right.clamp(-1, 1), align_corners=True).squeeze(-1)
    v_up    = F.grid_sample(vel, grid_up.clamp(-1, 1), align_corners=True).squeeze(-1)
    
    dv_dx = (v_right - v_center) / eps
    dv_dz = (v_up - v_center) / eps
    
    vel_mean = torch.mean(vel, dim=[2, 3]).view(b1,-1) # [b1, 1, 1, 1]
    vel_std = torch.std(vel, dim=[2, 3]).view(b1,-1)  # [b1, 1, 1, 1]    
    vel_feature = vel_std * 1.5 + vel_mean
    vel_feature = vel_feature.unsqueeze(1).expand(-1,b_pts,-1).permute(0, 2, 1)
    physical_context = torch.cat([vel_feature,dv_dx, dv_dz], dim=1).permute(0, 2, 1)
    # print('physical_context', physical_context.shape)
    # 返回的是 detach 后的特征，它作为 Trunk 的“静态输入”
    return physical_context
    
def generate_weight(vel, type="energy"):
    """
    vel: [B, 1, H, W] 或 [B, N_pts, 1]
    """
    if type == "energy":
        # 权重与 k^2 成正比，即与 1/v^2 成正比
        w = 1.0 / (vel**2 + 1e-6)
    elif type == "gradient":
        # 关注速度变化剧烈的区域（界面）
        # 仅适用于 2D 图像格式的 vel
        grad_x = torch.abs(vel[:, :, 1:, :] - vel[:, :, :-1, :])
        # ... 适当 padding 并取模长
        w = torch.nn.functional.pad(grad_x, (0, 0, 1, 0)) 
    
    # 归一化，保证积分均值为 1，避免 Loss 量级爆炸
    w = w / torch.mean(w, dim=(1, 2, 3) if w.dim()==4 else 1, keepdim=True)
    return w
def get_helmholtz_spatial_weights(coords, velocity, omega, source_loc, alpha=0.1, sigma=0.5):
    """
    计算方案一：物理尺度变权重项 lambda(x)
    
    Args:
        coords (torch.Tensor): 空间坐标, 形状为 (N, dim), 如 (N, 2) 或 (N, 3)
        velocity (torch.Tensor): 对应坐标处的速度模型值, 形状为 (N, 1)
        omega (float): 角频率 (2 * pi * f)
        source_loc (torch.Tensor): 震源位置坐标, 形状为 (1, dim)
        alpha (float): 波数平衡系数，调节高波数区域的惩罚强度
        sigma (float): 震源加权的高斯标准差，控制“能量锚点”的范围
        
    Returns:
        weights (torch.Tensor): 空间权重矩阵, 形状为 (N, 1)
    """
    
    # 1. 计算局部波数 k = omega / v
    k = omega / (velocity + 1e-6)
    k_sq = k**2
    
    # 2. 归一化波数项 (防止不同量级速度模型导致权重失效)
    # 使用均值或最大值归一化，使 k_bar^2 在 1 附近波动
    k_sq_bar = k_sq / torch.mean(k_sq)
    
    # 3. 计算算子平衡项: 1 / (1 + alpha * k_bar^2)
    # 逻辑：在 k 较大的区域（低速区，波长短，震荡剧烈），降低权重以防梯度爆炸
    operator_term = 1.0 / (1.0 + alpha * k_sq_bar)
    # print('k', operator_term.shape)
    source_loc = source_loc.unsqueeze(0).expand(coords.shape[0],coords.shape[1],-1)
    # 4. 计算源项锚点（高斯加权）
    # dist_sq: 每个点到震源的欧式距离平方
    dist_sq = torch.sum((coords - source_loc)**2, dim=-1)
    source_anchor = torch.exp(-dist_sq / (2 * sigma**2))
    # print('source_anchor', source_anchor.shape)
    # 5. 组合权重
    # 我们希望：在源附近权重极大（强制收敛），在全域受算子平衡项调节
    # 这里加 1e-2 是为了保证全域仍有基本的 PDE 约束，不至于震源之外完全不更新
    weights = operator_term * (source_anchor + 1e-2)
    
    return weights
    
def get_available_gpus(min_memory_gb=19):
    """结合nvidia-smi命令获取更准确的GPU内存信息"""
    import subprocess
    import re
    
    min_memory_bytes = min_memory_gb * 1024 * 1024 * 1024
    available_gpus = []
    
    try:
        # 调用nvidia-smi获取GPU内存信息
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        
        for line in result.strip().split("\n"):
            idx, total, free = map(int, re.findall(r"\d+", line))
            # 转换为字节
            free_bytes = free * 1024 * 1024
            if free_bytes >= min_memory_bytes:
                available_gpus.append(str(idx))
                
        return available_gpus
    except Exception as e:
        print(f"获取GPU信息失败，使用备用方法: {e}")
        # fallback到PyTorch的检测方法
        if not torch.cuda.is_available():
            return []
            
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            total_memory = prop.total_memory
            try:
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory = total_memory - allocated_memory
            except:
                free_memory = total_memory * 0.8
                
            if free_memory >= min_memory_bytes:
                available_gpus.append(str(i))
        
        return available_gpus
        
