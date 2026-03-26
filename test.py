import os
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 导入模块 (已移除 FNO)
# ==========================================
from model.utils import *
from model.dataloader import *
from model.PI_DeepOnet import Pi_DeepONet
from model.ploting import *

class Args_test:
    # ==========================================
    # 1. 路径与文件配置
    # ==========================================
    load_path = '/home/sharedata/zdg'         
    weights_save_path = '/home/sharedata/zdg' 
    save_doc = 'output_test'                       
    filename = 'PI_DeepONet_pde'              
    
    ext_val_datasets = {
        'Marmousi': {'prefix': 'marmousi_', 'loc_target': [0, 1, 2, 3, 4]}, 
    }
    
    # ==========================================
    # 2. 硬件与设备配置
    # ==========================================
    device = 2                                

    # ==========================================
    # 3. 学习率调度器参数
    # ==========================================
    factor = 0.9                              
    patience = 30                             
    min_lr = 1e-7                             
    
    # ==========================================
    # 4. 训练控制与状态保存
    # ==========================================
    if_load_model = False                     
    if_adjust = True                          
    adjust_from = 2000                        
    adjust_every = 1000                       
    adjust_speed = 1.1                        
    save_fig_every = 50                       
    save_model_every = 1000                    
    
    # ==========================================
    # 5. 数据集与批处理配置
    # ==========================================
    nvel_train = 1                         
    ny_train = 4900                           
    batch_size = 700                          
    batch_size_v = 1                           
    
    valid_rate = 0.1                          
    validate_every = 100                      
    valid_batch_size = 350                    
    valid_batch_size_v = 6                    
    accumulation_steps = 2                    

    source_list = [0, 1, 2, 3, 4]
    
    # ==========================================
    # 6. 物理网格与边界条件
    # ==========================================
    nx = 70                                   
    nz = 70                                   
    pml = True                                
    Lpml = 9                                  
    LD = 10 - Lpml                            
    
    # ==========================================
    # 7. 微调与域适应配置
    # ==========================================
    if_finetune = True                        
    ft_NIter = 1000                             
    ft_lr = 2e-5                              
    ft_a = 0.2                                
    ft_b = 1                                  
    ft_c = 1                                  
    
    # ==========================================
    # 8. 主训练循环超参数
    # ==========================================
    NIter = 10000 + 1                         
    warmup_epochs = 100                       
    lr = 1 * 1e-4                             
    weight_decay = 1e-4                       
    
    # ==========================================
    # 9. 物理信息损失函数权重
    # ==========================================
    a = 1                                     
    b = 1                                     
    c = 0                                     
    
    # ==========================================
    # 10. 网络架构与张量形状占位
    # ==========================================
    in_channels = 2                           
    in_channels_vel = 1                       
    input_shape_trunk = (batch_size, in_channels, 1, 2)       
    input_shape_branch1 = (batch_size, in_channels_vel, nz, nx) 
    input_shape_branch2 = (batch_size, in_channels, nz, nx)     

# ==============================================================================
# 核心数据提取与画图函数
# ==============================================================================

def extract_single_model_multi_source(args, vel_set, UU0_set, labels_set, base_count, target_model_idx=0):
    """提取单模型多震源数据"""
    num_sources = len(args.source_list)
    if target_model_idx >= base_count or target_model_idx < 0:
        raise ValueError(f"指定的索引 {target_model_idx} 超出范围，该集合只有 {base_count} 个基础模型。")
    
    vel_single = vel_set[target_model_idx].unsqueeze(0)
    UU0_list = []
    labels_list = []
    
    for s in range(num_sources):
        target_idx = target_model_idx + s * base_count
        UU0_list.append(UU0_set[target_idx].unsqueeze(0))
        # Label 现在形状为 [1, 2, Z, X]
        labels_list.append(labels_set[target_idx].unsqueeze(0))
        
    return {
        "vel": vel_single,
        "UU0_list": UU0_list,
        "labels_list": labels_list
    }

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_single_velocity_multi_sources(args, model, vel, UU0_list, labels_list, y_full_grid, epoch, save_doc, filename_prefix="MultiSource"):
    """
    针对单一速度模型，分别绘制实部和虚部的多震源波场预测对比图。
    优化 1：引入分 Batch 预测逻辑，防止测试大网格时 OOM。
    优化 2：调整 figsize 与 aspect='equal'，使子图呈现正方形。
    """
    model.eval()
    num_sources = len(UU0_list)
    nz, nx = vel.shape[2], vel.shape[3]
    
    # 获取 PML 裁剪边界
    L = args.LD
    slc = slice(L, -L) if L > 0 else slice(None)
    
    # === 修改这里：调整 figsize，让子图接近正方形 ===
    # 宽度 = 3.5 * 震源数，高度 = 10 (适应 3 行的比例)
    fig_real, axes_real = plt.subplots(3, num_sources, figsize=(3.5 * num_sources, 10))
    fig_imag, axes_imag = plt.subplots(3, num_sources, figsize=(3.5 * num_sources, 10))
    
    if num_sources == 1: 
        axes_real = axes_real[:, np.newaxis]
        axes_imag = axes_imag[:, np.newaxis]
        
    fig_real.suptitle(f"{filename_prefix} - REAL Part ({num_sources} Sources) | Epoch {epoch}", fontsize=16)
    fig_imag.suptitle(f"{filename_prefix} - IMAG Part ({num_sources} Sources) | Epoch {epoch}", fontsize=16)
    
    print("="*60)
    print(f"模型预测性能指标汇总：{filename_prefix}")
    print("="*60)
    
    with torch.no_grad():
        for s_idx in range(num_sources):
            curr_UU0 = UU0_list[s_idx].unsqueeze(0) if UU0_list[s_idx].dim() == 3 else UU0_list[s_idx]
            curr_label = labels_list[s_idx]
            
            # -----------------------------------------------------
            # 核心优化：分 Batch 预测网格点，防止显存爆炸
            # -----------------------------------------------------
            u_pred_list = []
            print(' curr_UU0',  curr_UU0.shape)
            # y_full_grid 的形状是 [1, B_pts, 2]，在 dim=1 上切分
            for y_batch in torch.split(y_full_grid, args.valid_batch_size, dim=1):
                # 推理当前的子网格
                u_batch = model(vel, y_batch, curr_UU0)
                u_pred_list.append(u_batch)
                
            # 将所有 batch 的预测结果在空间点维度拼回全图
            pred_concat = torch.cat(u_pred_list, dim=1)
            pred_np = pred_concat[0].view(nz, nx, 2).cpu().numpy()
            
            # 提取预测值并裁掉 PML 边界
            pred_real = pred_np[slc, slc, 0]
            pred_imag = pred_np[slc, slc, 1]
            
            # 兼容处理 Label 形状并提取真实值 (裁掉 PML)
            if curr_label.dim() == 4:
                if curr_label.shape[1] == 2: # [1, 2, Z, X]
                    true_real = curr_label[0, 0, :, :].cpu().numpy()[slc, slc]
                    true_imag = curr_label[0, 1, :, :].cpu().numpy()[slc, slc]
                else: # [1, Z, X, 2]
                    true_real = curr_label[0, :, :, 0].cpu().numpy()[slc, slc]
                    true_imag = curr_label[0, :, :, 1].cpu().numpy()[slc, slc]
            elif curr_label.dim() == 3 and curr_label.shape[-1] == 2:
                true_real = curr_label[:, :, 0].cpu().numpy()[slc, slc]
                true_imag = curr_label[:, :, 1].cpu().numpy()[slc, slc]
            else:
                true_real = curr_label[0, :, 0].view(nz, nx).cpu().numpy()[slc, slc]
                true_imag = curr_label[0, :, 1].view(nz, nx).cpu().numpy()[slc, slc]
            
            # 计算原始残差与定量评估指标
            err_real = true_real - pred_real
            err_imag = true_imag - pred_imag
            
            metrics_real = calculate_regression_metrics(pred_real, true_real)
            metrics_imag = calculate_regression_metrics(pred_imag, true_imag)
            
            print(f"\n---> Source {s_idx+1} <---")
            print(f"【实部】 MSE: {metrics_real['mse']:.6e} | MAE: {metrics_real['mae']:.6f} | R²: {metrics_real['r2']:.6f}")
            print(f"【虚部】 MSE: {metrics_imag['mse']:.6e} | MAE: {metrics_imag['mae']:.6f} | R²: {metrics_imag['r2']:.6f}")
            
            vmax_r, vmin_r = np.max(np.abs(true_real)), -np.max(np.abs(true_real))
            vmax_i, vmin_i = np.max(np.abs(true_imag)), -np.max(np.abs(true_imag))
            
            eRr = np.max(np.abs(err_real))
            eRi = np.max(np.abs(err_imag))
            
            # ================= [实部绘制 fig_real] =================
            # 修改了 aspect='equal' 保证正方形
            ax = axes_real[0, s_idx]
            im = ax.imshow(true_real, cmap='seismic', aspect='equal', vmin=vmin_r, vmax=vmax_r)
            ax.set_title(f"Src {s_idx+1} True Real\nR²: {metrics_real['r2']:.4f}", fontsize=11)
            ax.axis('off')
            fig_real.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = axes_real[1, s_idx]
            im = ax.imshow(pred_real, cmap='seismic', aspect='equal', vmin=vmin_r, vmax=vmax_r)
            ax.set_title(f"Src {s_idx+1} Pred Real", fontsize=11)
            ax.axis('off')
            fig_real.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = axes_real[2, s_idx]
            im = ax.imshow(err_real, cmap='bwr', aspect='equal', vmin=-eRr, vmax=eRr)
            ax.set_title(f"Src {s_idx+1} Err Real\nMSE: {metrics_real['mse']:.2e}", fontsize=11)
            ax.axis('off')
            fig_real.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # ================= [虚部绘制 fig_imag] =================
            ax = axes_imag[0, s_idx]
            im = ax.imshow(true_imag, cmap='seismic', aspect='equal', vmin=vmin_i, vmax=vmax_i)
            ax.set_title(f"Src {s_idx+1} True Imag\nR²: {metrics_imag['r2']:.4f}", fontsize=11)
            ax.axis('off')
            fig_imag.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = axes_imag[1, s_idx]
            im = ax.imshow(pred_imag, cmap='seismic', aspect='equal', vmin=vmin_i, vmax=vmax_i)
            ax.set_title(f"Src {s_idx+1} Pred Imag", fontsize=11)
            ax.axis('off')
            fig_imag.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax = axes_imag[2, s_idx]
            im = ax.imshow(err_imag, cmap='bwr', aspect='equal', vmin=-eRi, vmax=eRi)
            ax.set_title(f"Src {s_idx+1} Err Imag\nMSE: {metrics_imag['mse']:.2e}", fontsize=11)
            ax.axis('off')
            fig_imag.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    # --- 调整布局并保存实部 ---
    fig_real.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_doc, exist_ok=True)
    save_path_real = os.path.join(save_doc, f"{filename_prefix}_REAL_epoch_{epoch}.png")
    fig_real.savefig(save_path_real, dpi=150, bbox_inches='tight') 
    plt.close(fig_real) 
    
    # --- 调整布局并保存虚部 ---
    fig_imag.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_imag = os.path.join(save_doc, f"{filename_prefix}_IMAG_epoch_{epoch}.png")
    fig_imag.savefig(save_path_imag, dpi=150, bbox_inches='tight') 
    plt.close(fig_imag) 
    
    print(f"\n[+] 实部对比图已保存: {save_path_real}")
    print(f"[+] 虚部对比图已保存: {save_path_imag}\n")

# ==============================================================================
# 核心测试流程 (完全去除 Dataloader 依赖的纯 Tensor 版本)
# ==============================================================================

def test(args, target_epoch, custom_weights_path=None):
    try:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        print(f"\n========== 开始测试评估模式 | 设备: {device} ==========")
        
        # ---------------------------------------------------------
        # 1. 基础数据准备与拆分 (纯 Tensor 操作)
        # ---------------------------------------------------------
        vel_original = load_tensor_from_npy(args.load_path, 'velocity_data_70_70_n1.npy')
        UU0_original = load_tensor_from_npy(args.load_path, 'backgroundfield_data_freq5_1source_70_70_n1.npy')
        UU_original = load_tensor_from_npy(args.load_path, 'wavefield_data_freq5_5sources_70_70_n1.npy')
        
        args.nx = args.nx + args.LD * 2
        args.nz = args.nz + args.LD * 2
        
        if args.pml:
            Lpml = args.Lpml
            vel = vel_original[:, Lpml:-Lpml, Lpml:-Lpml]
            UU0 = UU0_original[:, :, Lpml:-Lpml, Lpml:-Lpml]
            UU = UU_original[:, :, Lpml:-Lpml, Lpml:-Lpml]
        else:
            vel, UU0, UU = vel_original, UU0_original, UU_original
    
        UU_loc = [UU[loc * len(vel) : (loc + 1) * len(vel), ...] for loc in range(5)]
        UU0_loc = [UU0[loc * len(vel) : (loc + 1) * len(vel), ...] for loc in range(5)]
        
        np.random.seed(1)
        vel_train, UU_loc_train, UU0_train, y_train, labels_train, \
        vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid = Training_data(args, vel, UU_loc, UU0_loc)
        print('y_valid', y_valid.shape)
        vel_train = vel_train / 1000.
        vel_valid = vel_valid / 1000.
        
        train_plot_data = extract_single_model_multi_source(
            args, vel_train, UU0_train, labels_train, base_count=args.nvel_train, target_model_idx=0
        )
        valid_plot_data = extract_single_model_multi_source(
            args, vel_valid, UU0_valid, labels_valid, base_count=(int(args.valid_rate * args.nvel_train) + 1), target_model_idx=0
        )

        # ---------------------------------------------------------
        # 2. 外部测试集准备 (Marmousi, 提取纯 Tensor)
        # ---------------------------------------------------------
        ext_val_sets = {}
        if hasattr(args, 'ext_val_datasets'):
            for name, config in args.ext_val_datasets.items():
                _, p_data = prepare_external_val_dataset(
                    args, 
                    prefix=config['prefix'], 
                    loc_target=config['loc_target'], 
                    y_pred_grid=y_valid.unsqueeze(0) 
                )
                ext_val_sets[name] = p_data
                
        # ---------------------------------------------------------
        # 3. 模型初始化与权重加载
        # ---------------------------------------------------------
        model = Pi_DeepONet(args).to(device)
        
        if custom_weights_path:
            model_path = custom_weights_path
        else:
            model_path = os.path.join(args.save_doc, f'{args.filename}_PI_model_{target_epoch}epoch_weights.pth')
            
        print(f"[*] 正在加载 PI_DeepONet 权重: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到指定的权重文件: {model_path}，请检查 target_epoch 是否正确。")
            
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        y_grid_tensor = y_valid.unsqueeze(0).to(device) 
        
        # ---------------------------------------------------------
        # 4. 可视化绘图与评估计算 (合并执行)
        # ---------------------------------------------------------
        print("\n========== 开始评估与生成可视化结果 ==========")

        plot_single_velocity_multi_sources(
            args,
            model=model,
            vel=valid_plot_data["vel"].to(device),
            UU0_list=[u.to(device) for u in valid_plot_data["UU0_list"]],
            labels_list=[l.to(device) for l in valid_plot_data["labels_list"]],
            y_full_grid=y_grid_tensor,
            epoch=target_epoch,
            save_doc=args.save_doc,
            filename_prefix="Valid_Set_Model"
        )

        plot_single_velocity_multi_sources(
            args,
            model=model,
            vel=train_plot_data["vel"].to(device),
            UU0_list=[u.to(device) for u in train_plot_data["UU0_list"]],
            labels_list=[l.to(device) for l in train_plot_data["labels_list"]],
            y_full_grid=y_grid_tensor,
            epoch=target_epoch,
            save_doc=args.save_doc,
            filename_prefix="Train_Set_Model"
        )
        
        if 'Marmousi' in ext_val_sets:
            mar_data = ext_val_sets['Marmousi']
            v_m_test = mar_data["v_test"].to(device)   
            u0_m_test = mar_data["u0_test"].to(device) 
            lab_m_test = mar_data["lab_test"].to(device) 
            
            num_mar_sources = u0_m_test.shape[0]
            mar_vel_single = v_m_test[0:1] 
            
            mar_UU0_list = [u0_m_test[i:i+1] for i in range(num_mar_sources)]
            mar_labels_list = [lab_m_test[i:i+1] for i in range(num_mar_sources)]
            
            plot_single_velocity_multi_sources(
                args,
                model=model,
                vel=mar_vel_single,
                UU0_list=mar_UU0_list,
                labels_list=mar_labels_list,
                y_full_grid=y_grid_tensor, 
                epoch=target_epoch,
                save_doc=args.save_doc,
                filename_prefix="External_Marmousi"
            )

        print(f"\n========== 测试流程完毕！可视化结果已保存至: {args.save_doc} ==========")

    except Exception as e:
        print(f"测试过程发生错误: {e}")
        raise
    finally:
        torch.cuda.empty_cache()
        gc.collect()

def main():
    args = Args_test()
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(args.device)}")
        min_gpu_memory = 23  # GB
        gpu_memory = torch.cuda.get_device_properties(args.device).total_memory / (1024 **3)
        if gpu_memory < min_gpu_memory:
            print(f"警告：GPU内存 {gpu_memory:.1f}GB 小于最小要求 {min_gpu_memory}GB，可能导致OOM")
    else:
        print("未找到可用GPU，将使用CPU训练")
    
    test(args, 500) 

if __name__ == "__main__":
    torch.cuda.empty_cache() 
    print('*******************************************')
    print('           START EVALUATION                ')
    print('*******************************************')
    main()