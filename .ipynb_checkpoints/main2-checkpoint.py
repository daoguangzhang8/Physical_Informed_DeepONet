from Labconfig import *
from config import *
from model.PI_DeepOnet import *
from model.train import train
from model.ploting import *



def main():
    # 参数设置
    args = Args()
    
    # 检查GPU可用性（单卡训练）
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        # 可选：检查GPU内存是否满足最小要求（根据实际需求调整）
        min_gpu_memory = 23  # GB
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 **3)  # 转换为GB
        if gpu_memory < min_gpu_memory:
            print(f"警告：GPU内存 {gpu_memory:.1f}GB 小于最小要求 {min_gpu_memory}GB，可能导致OOM")
    else:
        print("未找到可用GPU，将使用CPU训练")
    
    # 直接调用单卡训练函数
    train(args)  # 注意：此处调用的是修改后的单卡版本train函数

if __name__ == "__main__":
    torch.cuda.empty_cache()  # 清空缓存
    print('*******************************************')
    print('           START TRAINING Pi_DeepONet      ')
    print('*******************************************')
    main()