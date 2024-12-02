import os
import json
import random

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from utils.file_client import FileClient
from utils.img_util import imfrombytes
from utils.flow_util import resize_flow, flowread
from core.utils import (create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip,GroupRandomHorizontalFlowFlip, GroupRandomHorizontalFlowFlipStereo, GroupRandomHorizontalFlipStereo)


class  TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']
        self.flow_root = args['flow_root']
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)
        
        json_path = os.path.join('./datasets', args['name'], 'train.json')

        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict.keys()))

        # self.video_names = sorted(os.listdir(self.video_root))
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list
                

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)
            masks.append(all_masks[idx])

            if len(frames) <= self.num_local_frames-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_local_frames and random.random() < 0.5:
                frames.reverse()
                masks.reverse()
                if self.load_flow:
                    flows_f.reverse()
                    flows_b.reverse()
                    flows_f, flows_b = flows_b, flows_f
        if self.load_flow:
            frames, flows_f, flows_b = GroupRandomHorizontalFlowFlip()(frames, flows_f, flows_b)
        else:
            frames = GroupRandomHorizontalFlip()(frames)

        # normalize, to tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        # img [-1,1] mask [0,1]
        if self.load_flow:
            return frame_tensors, mask_tensors, flows_f, flows_b, video_name
        else:
            return frame_tensors, mask_tensors, 'None', 'None', video_name


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args['size']

        self.video_root = args['video_root']
        self.mask_root = args['mask_root']
        self.flow_root = args['flow_root']

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)
        self.video_names = sorted(os.listdir(self.mask_root))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            self.video_dict[v] = v_len
            self.frame_dict[v] = frame_list

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        selected_index = list(range(self.video_dict[video_name]))

        # read video frames
        frames = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            frame_path = os.path.join(self.video_root, video_name, frame_list[idx])

            img_bytes = self.file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)

            mask_path = os.path.join(
                self.mask_root, video_name, f'{str(idx).zfill(5)}.png'
            )
            mask = Image.open(mask_path).resize(self.size, Image.NEAREST).convert('L')

            # origin: 0 indicates missing. now: 1 indicates missing
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)

            m = cv2.dilate(m,
                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                           iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

            if len(frames) <= len(selected_index)-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

        # normalize, to tensors
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)

        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        if self.load_flow:
            return frame_tensors, mask_tensors, flows_f, flows_b, video_name, frames_PIL
        else:
            return frame_tensors, mask_tensors, 'None', 'None', video_name
        
        
class TrainDatasetStereo(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']  # ../dataset/Monkaa/
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        self.load_flow = args['load_flow']

        # 视频字典结构
        self.video_dict = {}  # 存储视频和对应帧路径
        self.frame_dict = {}  # 存储帧的文件列表

        for scene in sorted(os.listdir(self.video_root)):
            scene_path = os.path.join(self.video_root, scene)
            for view in ['left', 'right']:
                view_path = os.path.join(scene_path, view, 'frames')
                if not os.path.exists(view_path):
                    continue
                frame_list = sorted(os.listdir(view_path))
                num_frames = len(frame_list)

                # 只有帧数足够的视频才会被添加
                if num_frames > self.num_local_frames + self.num_ref_frames:
                    video_key = f"{scene}/{view}"
                    self.video_dict[video_key] = num_frames
                    self.frame_dict[video_key] = frame_list

        # 获取所有视频的键值
        self.video_names = list(self.video_dict.keys())

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
        scene, view = video_name.split('/')
        masks_path = os.path.join(self.video_root, scene, view, 'masks')
        wrapped_frames_path = os.path.join(self.video_root, scene, view, 'wrapped_frames')
        if view == 'left':
            another_view = 'right'
        elif view == 'right':
            another_view = 'left'
        gt_frames_path = os.path.join(self.video_root, scene, another_view, 'frames')
        flows_path = os.path.join(self.video_root, scene, another_view, 'flow')

        # 抽取索引
        num_frames = self.video_dict[video_name]
        selected_index = self._sample_index(num_frames, self.num_local_frames, self.num_ref_frames)

        # 加载帧、掩码和光流
        gt_frames = []
        wrapped_frames = []
        masks = []
        flows_f, flows_b = [], []

        for idx in selected_index:
            frame_name = self.frame_dict[video_name][idx]
            img_path = os.path.join(gt_frames_path, frame_name)
            mask_path = os.path.join(masks_path, frame_name)
            wrapped_frame_path = os.path.join(wrapped_frames_path, frame_name)  # wrapped_frames 路径
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)
            
            # 加载 wrapped_frame
            wrapped_img_bytes = self.file_client.get(wrapped_frame_path, 'img')
            wrapped_img = imfrombytes(wrapped_img_bytes, float32=False)
            wrapped_img = cv2.cvtColor(wrapped_img, cv2.COLOR_BGR2RGB)
            wrapped_img = cv2.resize(wrapped_img, self.size, interpolation=cv2.INTER_LINEAR)
            wrapped_img = Image.fromarray(wrapped_img)

            mask = Image.open(mask_path).convert('L').resize(self.size, Image.NEAREST)

            gt_frames.append(img)
            wrapped_frames.append(wrapped_img)  # 添加 wrapped_frame
            masks.append(mask)

            if len(gt_frames) <= self.num_local_frames - 1 and self.load_flow:
                current_frame = frame_name[:-4]
                next_frame = self.frame_dict[video_name][idx + 1][:-4]
                flow_f_path = os.path.join(flows_path, f"{current_frame}_{next_frame}_f.flo")
                flow_b_path = os.path.join(flows_path, f"{next_frame}_{current_frame}_b.flo")
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)
            
            # local frames 随机时序反向
            if len(gt_frames) == self.num_local_frames and random.random() < 0.5:
                gt_frames.reverse()
                wrapped_frames.reverse()  # 同时反向 wrapped_frames
                masks.reverse()
                if self.load_flow:
                    flows_f.reverse()
                    flows_b.reverse()
                    flows_f, flows_b = flows_b, flows_f

        # 数据增强
        if self.load_flow:
            gt_frames, wrapped_frames, masks, flows_f, flows_b = GroupRandomHorizontalFlowFlipStereo()(gt_frames, wrapped_frames, masks, flows_f, flows_b)
        else:
            gt_frames, wrapped_frames, masks = GroupRandomHorizontalFlipStereo()(gt_frames, wrapped_frames, masks)

        # 转换为张量
        frame_tensors = self._to_tensors(gt_frames) * 2.0 - 1.0
        wrapped_frame_tensors = self._to_tensors(wrapped_frames) * 2.0 - 1.0  # wrapped_frames 转换为张量
        mask_tensors = self._to_tensors(masks)
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1)  # H, W, 2, T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        # 返回张量
        if self.load_flow:
            return frame_tensors, wrapped_frame_tensors, mask_tensors, flows_f, flows_b, video_name
        else:
            return frame_tensors, wrapped_frame_tensors, mask_tensors, None, None, video_name