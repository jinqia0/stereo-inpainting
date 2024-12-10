import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from core.dataset import TestDatasetStereo

# 假设 TestDatasetStereo 已经被定义
args = {
    'size': (1432, 840),  # 输入尺寸
    'video_root': '/home/jinqiao/stereo-inpainting/Datasets/Kitti/test',  # 数据集根目录
    'load_flow': False,  # 暂时关闭光流加载以简化可视化
    'current_view': None,  # 设置 None 则随机选择 left 或 right
}

# 初始化数据集和数据加载器
dataset = TestDatasetStereo(args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 从数据集中取一组样本
for batch in dataloader:
    frame_tensors, mask_tensors, flows_f, flows_b, video_name, opposite_frames_PIL = batch

    # 转换张量为 NumPy 数组
    frame_images = frame_tensors[0].permute(0, 2, 3, 1).numpy()  # T H W C
    mask_images = mask_tensors[0].squeeze(1).numpy()  # T H W

    # 可视化结果
    num_frames_plt = 1

    plt.figure(figsize=(15, 5 * num_frames_plt))
    for i in range(num_frames_plt):
        # 当前视角帧
        plt.subplot(3, num_frames_plt, i * 3 + 1)
        plt.imshow((frame_images[i] + 1) / 2)  # 归一化到 [0, 1]
        plt.title(f"Frame {i} - Current View")
        plt.axis('off')

        # 掩码图像
        plt.subplot(3, num_frames_plt, i * 3 + 3)
        plt.imshow(mask_images[i], cmap='gray')
        plt.title(f"Mask {i}")
        plt.axis('off')

        # 对视角帧
        plt.subplot(3, num_frames_plt, i * 3 + 2)
        plt.imshow(opposite_frames_PIL[i].squeeze(0))
        plt.title(f"Frame {i} - Opposite View")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig("output.png")
    break  # 仅显示一个批次
