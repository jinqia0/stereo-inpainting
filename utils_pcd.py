import numpy as np
import open3d as o3d
import torch
import pywt
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
from torchvision import transforms


def create_view_transform_matrix(d, theta_degrees, view='left'):
    """
    创建从全局坐标系到指定眼睛坐标系的转换矩阵，假设绕Y轴旋转。
    """
    theta_radians = np.radians(theta_degrees)
    translation_distance = -d / 2 if view == 'right' else d / 2
    rotation_angle = -theta_radians if view == 'right' else theta_radians

    translation_matrix = np.array([
        [1, 0, 0, translation_distance],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])

    transform_matrix = np.dot(rotation_matrix, translation_matrix)
    return transform_matrix


def apply_transform_to_point(point, transform_matrix):
    """
    应用转换矩阵到一个点上。
    """
    point_homogeneous = np.hstack((point, np.ones((point.shape[0], 1)))).T
    transformed_point_homogeneous = np.dot(transform_matrix, point_homogeneous)
    transformed_point = transformed_point_homogeneous[:3] / transformed_point_homogeneous[3]
    return transformed_point.T


def piecewise_function(x, depth_exp=1):
    # 创建一个与 x 形状相同的数组，用于存储结果
    y = np.zeros_like(x)
    
    # 对于 x > 0.5 的情况，y = 2x
    y[x > 0.5] =  1- 0.5 * (((1 - x[x > 0.5]) / 0.5)**depth_exp)
    
    # 对于 x <= 0.5 的情况，y = x
    y[x <= 0.5] = 0.5 * (((x[x <= 0.5]) / 0.5)**depth_exp) 
    
    return y


def depth_norm(depth_numpy, depth_exp=1, wave_coeff=False):
    depth_max, depth_min = depth_numpy.max(), depth_numpy.min()
    depth_numpy = (depth_numpy-depth_min)/(depth_max-depth_min)
    if wave_coeff:
        # 进行二维离散小波变换
        coeffs2 = pywt.dwt2(depth_numpy, 'haar')
        # 提取变换系数
        LL, (LH, HL, HH) = coeffs2
        # 仅保留低频分量LL，其他分量置零
        LH[:] = 0
        HL[:] = 0
        HH[:] = 0
        # 重构图像
        depth_numpy = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    
    depth_numpy = depth_numpy**depth_exp
    depth_numpy = piecewise_function(depth_numpy, depth_exp=depth_exp)
    depth_numpy = depth_numpy*(depth_max-depth_min)+depth_min
    return depth_numpy


def get_rgb_from_view(
        rgb_image, depth_image, device, view='right', d_eye=0.10, theta_degrees=0,
        if_show=False, amp_factor=10):
    """
    GPU加速的从特定视角获取RGB函数。

    参数:
    - color_path (str): 输入图像路径。
    - depth_tensor (torch.Tensor): 深度图张量，值范围为 [0, 1]。
    - device (str): 计算设备，如 'cuda' 或 'cpu'。
    - view (str): 左眼或右眼视角 ('left' 或 'right')。
    - d_eye (float): 双眼间距。
    - theta_degrees (float): 旋转角度。
    - if_show (bool): 是否显示可视化结果。

    返回:
    - 重新投影后的图像。
    - 点云对象。
    """
    # 读取彩色图像
    color_numpy = np.array(rgb_image).astype(np.float32)
    depth_numpy = np.array(depth_image).astype(np.float32)  # 将深度张量转换为 NumPy 数组
    H, W, _ = color_numpy.shape

    # depth_numpy 预处理
    eps = 1e-2
    depth_max, depth_min = depth_numpy.max(), depth_numpy.min()
    depth_numpy = (depth_numpy-depth_min) / (depth_max-depth_min)
    depth_numpy = 1 / (depth_numpy + eps)
    
    # 转换图像和深度为 Open3D 的张量格式
    color = o3d.t.geometry.Image(color_numpy).to(device)
    depth = o3d.t.geometry.Image(depth_numpy).to(device)

    # 创建 RGBD 图像
    rgbd = o3d.t.geometry.RGBDImage(color, depth)

    # 计算相机内参
    focal_length_in_mm, sensor_width_in_mm = 24, 36
    fx = (focal_length_in_mm / sensor_width_in_mm) * W
    fy = fx
    intrinsic = o3d.core.Tensor([[fx, 0, W / 2.0],
                                 [0, fy, H / 2.0],
                                 [0.0, 0.0, 1.0]]).to(device)

    # 使用 GPU 加速创建点云
    eps = 1e-2
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic, depth_scale=1.0, depth_max=1.0 / eps + 10,
        stride=1, with_normals=True)

    # 应用人眼的外参矩阵调整点云坐标
    view_matrix = create_view_transform_matrix(d_eye, theta_degrees, view=view)
    pcd.point['positions'] = o3d.core.Tensor(
        apply_transform_to_point(pcd.point['positions'].cpu().numpy(), view_matrix),
        dtype=o3d.core.Dtype.Float32).to(device)

    # 将点云重新投影到视角平面
    rgbd_reproj = pcd.project_to_rgbd_image(
        W, H, intrinsic, depth_scale=1.0, depth_max=1.0 / eps + 10)

    if if_show:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        # 如果范围不在 [0, 1] 或 [0, 255]，进行归一化
        if color_numpy.max() > 1.0:  # 假设是 [0, 255]，则归一化到 [0, 1]
            color_numpy /= 255.0

        axs[0][0].imshow(color_numpy)
        axs[0][0].set_title('RGB Original')

        color_reproj = np.asarray(rgbd_reproj.color.to_legacy())
        # 可视化时确保值在 [0, 1] 范围
        if color_reproj.max() > 1.0:
            color_reproj /= 255.0
        axs[0][1].imshow(color_reproj)
        axs[0][1].set_title(f'Reprojected {view} RGB')

        depth_ori = 1 / depth_numpy - eps
        axs[1][0].imshow((depth_ori - depth_ori.min()) / (depth_ori.max() - depth_ori.min() + eps), cmap='inferno')
        axs[1][0].set_title('Original Depth')

        depth_reproj = 1 / (np.asarray(rgbd_reproj.depth.to_legacy()) + eps)
        depth_reproj = ((depth_reproj - depth_ori.min()) / (depth_ori.max() - depth_ori.min() + eps)).clip(0.0, 1.0)
        axs[1][1].imshow(depth_reproj, cmap='inferno')
        axs[1][1].set_title(f'Reprojected {view} Depth')
        plt.show()

    return np.asarray(rgbd_reproj.color.to_legacy())


if __name__ == '__main__':    
    # 输入图像路径
    image_path = '/home/qiao/T-SVG/Monkaa/a_rain_of_stones_x2/left/0000.png'
    
    # 加载和调整图像大小
    transform_resize = transforms.Resize((240, 432))
    image = Image.open(image_path).convert("RGB")
    
    # 深度估计模型
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    depth_image = pipe(image)['depth']
    
    rgb_image = transform_resize(image)  # 调整到 (432, 240)
    depth_image = transform_resize(depth_image)
    
    # 调用 `get_rgb_from_view`
    rgb_out = get_rgb_from_view(rgb_image, depth_image, if_show=False, device=o3d.core.Device("CPU:0"))

    cv2.imwrite('/home/qiao/T-SVG/test.png', cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))
