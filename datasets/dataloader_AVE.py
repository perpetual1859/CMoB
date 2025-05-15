# csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/test.csv'  # 完整路径
# self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
# self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'
#CREMAD  6类    AVE  28类
import os
from PIL import Image
import pickle
from numpy.ma import copy
from torchvision import transforms
import numpy as np
import random
import torch
from torch.utils.data import Dataset

# 标签映射 CREMAD
# LABEL_MAP = {"NEU": 0, "HAP": 1, "ANG": 2, "SAD": 3, "FEA": 4, "DIS": 5}
#AVE



class AV_KS_Dataset(Dataset):
    def __init__(self, mode, transforms=None):
        self.data = []
        self.label = []
        classes =[]
        # # 根据模式选择路径 CEMA_D
        # if mode == 'train':
        #     csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/train.csv'  # 完整路径
        #     self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
        #     self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'
        # elif mode == 'val':
        #     csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/val.csv'  # 完整路径
        #     self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
        #     self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'
        # else:
        #     csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/test.csv'  # 完整路径
        #     self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
        #     self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'

        # 根据模式选择路径 AVE
        if mode == 'train':
            csv_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/trainSet.txt'  # 完整路径
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS'
            # self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS-SE'
            # self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Image-01-FPS-SE'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/AVE-flow'
        elif mode == 'val':
            csv_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/valSet.txt'  # 完整路径
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS'
            # self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS-SE'
            # self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Image-01-FPS-SE'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/AVE-flow'
        else:
            csv_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/testSet.txt'  # 完整路径
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS'
            # self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS-SE'
            # self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Image-01-FPS-SE'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/AVE-flow'



        # # 读取CSV文件并加载数据，第一列为文件名，第二列为标签
        # with open(csv_path) as f:
        #     for line in f:
        #         item = line.strip().split(",")  # 使用逗号分隔，一行中所有元素
        #         name = item[0]  # 提取文件名
        #         label = item[1]  # 提取标签
        #
        #         # 检查音频文件是否存在
        #         audio_file = os.path.join(self.audio_path, name + '.npy')
        #         if os.path.exists(audio_file):
        #             self.data.append(name)
        #             self.label.append(LABEL_MAP[label])  # 将标签映射为数值，自己定义标签映射

                # 读取数据 确定类别
        with open(csv_path, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i
        # print(len(class_dict))
        # 读取数据
        with open(csv_path, 'r') as f2:
            files = f2.readlines()
            for item in files:
                item = item.split('&')
                name = item[1]
                label = item[0]
                audio_path = os.path.join(self.audio_path, name + '.npy')

                # visual_path = os.path.join(self.visual_path, 'Image-{:02d}-FPS-SE'.format(self.args.fps), item[1])

                # if os.path.exists(audio_path) and os.path.exists(visual_path):
                if os.path.exists(audio_path):
                    # if audio_path not in self.audio:
                    # self.image.append(visual_path)
                    self.data.append(name)
                    self.label.append(class_dict[label])
                else:
                    continue
        # print('Data load finished')
        # length = len(self.data)

        print(f'Data load finished. Number of files: {len(self.data)}')

        self.mode = mode
        self.transforms = transforms
        self._init_atransform()

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        # 加载音频数据
        spectrogram = np.load(os.path.join(self.audio_path, av_file + '.npy'),allow_pickle=True)
        spectrogram = np.expand_dims(spectrogram, axis=0)


        # 图像数据加载路径
        path = os.path.join(self.visual_path, av_file)  # 图像文件夹路径
        # file_num = len([lists for lists in os.listdir(path)])

        image_files = [file for file in os.listdir(path) if file.endswith('.jpg')]  # 获取所有jpg文件
        file_num = len(image_files)

        # 图像数据增强
        if self.mode == 'train':
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        #原代码
        # pick_num = 3    #指定要从文件中选择的图像数量。
        # seg = int(file_num / pick_num) #计算每个选择段中的图像数量（假设file_num是总图像数量）
        # path1 = []  #存储每个选中图像的路径。
        # image = []  #存储打开的图像对象。
        # image_arr = []  #存储转换后的图像数据（可能是为了机器学习模型准备的）
        # t = [0] * pick_num  #一个列表，用于存储每个选中图像的索引。
        #
        # for i in range(pick_num):
        #     if self.mode == 'train':
        #         t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
        #         if t[i] >= 10:
        #             t[i] = 9
        #     else:
        #         t[i] = i * seg + max(int(seg / 2), 1) if file_num > 6 else 1
        #
        #     path1.append('frame_0000' + str(t[i]) + '.jpg')
        #     image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
        #
        #     image_arr.append(transf(image[i]))
        #     image_arr[i] = image_arr[i].unsqueeze(1).float()
        #     if i == 0:
        #         image_n = copy.copy(image_arr[i])
        #     else:
        #         image_n = torch.cat((image_n, image_arr[i]), 1)
        #
        # label = self.label[idx]
        #
        # return image_n, spectrogram, label, idx
        #



        ##########################
        #修改后的代码
        # 如果图片数少于 3 张，则使用所有图片
        if len(image_files) < 3:
            selected_images = image_files  # 使用所有图片
        else:
            # 随机选择 3 张图片
            # file_num=len(image_files)
            segment_size=file_num // 3
            segments = [
                (i * segment_size, (i + 1) * segment_size) for i in range(3)
            ]

            # 在每个段内随机选择一个索引
            selected_indices = [random.randint(segment[0], segment[1] - 1) for segment in segments]

            # 使用这些索引从图像文件列表中选取图片
            selected_images = [image_files[i] for i in selected_indices]
            # selected_images = random.sample(image_files, 3)

        # 图像数据增强
        image_arr = []
        for img_name in selected_images:
            img_path = os.path.join(path, img_name)
            image = Image.open(img_path)
            # image = Image.open(img_path).convert('RGB')
            image_arr.append(transf(image).unsqueeze(0).float())

        # 如果图像数量不足 3 张，补充空的图像（例如：全零图像）
        if len(image_arr) < 3:
            # 使用一个全零的默认图像来填补
            default_image = torch.zeros_like(image_arr[0])  # 创建一个与原图像相同尺寸的零图像
            while len(image_arr) < 3:
                image_arr.append(default_image)

        # 合并所有选择的图像
        # image_n = torch.cat(image_arr, dim=1)
        # 现在 image_arr 包含了 3 张随机选择的图像
        # 合并所有选择的图像
        image_n = torch.cat(image_arr, dim=0)

        label = self.label[idx]  # 获取标签

        return image_n, spectrogram, label, idx


class AV_KS_Dataset_sample_level(Dataset):
    def __init__(self, mode, contribution, transforms=None):
        self.data = []
        self.label = []
        self.drop = []
        classes = []

        # # 根据模式选择路径 CEMA_D
        # if mode == 'train':
        #     csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/train.csv'  # 完整路径
        #     self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
        #     self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'
        # elif mode == 'val':
        #     csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/val.csv'  # 完整路径
        #     self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
        #     self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'
        # else:
        #     csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/test.csv'  # 完整路径
        #     self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
        #     self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'

        # 根据模式选择路径 AVE

        if mode == 'train':
            csv_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/trainSet.txt'  # 完整路径
            # self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS-SE'
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS'
            # self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Image-01-FPS-SE'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/AVE-flow'
        elif mode == 'val':
            csv_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/valSet.txt'  # 完整路径
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS'
            # self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS-SE'
            # self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Image-01-FPS-SE'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/AVE-flow'
        else:
            csv_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/testSet.txt'  # 完整路径
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS'
            # self.audio_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Audio-01-FPS-SE'
            # self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/Image-01-FPS-SE'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/AVE_Dataset/AVE-flow'


            # 读取CSV文件并加载数据
        # with open(csv_path) as f:
        #     for line in f:
        #         item = line.strip().split(",")  # 使用逗号分隔
        #         name = item[0]  # 提取文件名
        #         label = item[1]  # 提取标签
        #
        #         # 检查音频文件是否存在
        #         audio_file = os.path.join(self.audio_path, name + '.npy')
        #         if os.path.exists(audio_file):
        #             self.data.append(name)
        #             self.label.append(LABEL_MAP[label])  # 将标签映射为数值
        #             self.drop.append(0)
        #
        # print('Data load finished')
        # length = len(self.data)

        #读取数据 确定类别
        with open(csv_path, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i
        #读取数据
        with open(csv_path, 'r') as f2:
            files = f2.readlines()
            for item in files:
                item = item.split('&')
                name = item[1]
                label = item[0]
                audio_path = os.path.join(self.audio_path, item[1] + '.npy')
                # visual_path = os.path.join(self.visual_path, 'Image-{:02d}-FPS-SE'.format(self.args.fps), item[1])

                # if os.path.exists(audio_path) and os.path.exists(visual_path):
                if os.path.exists(audio_path) :
                    # if audio_path not in self.audio:
                        # self.image.append(visual_path)
                        self.data.append(name)
                        self.label.append(class_dict[item[0]])
                        self.drop.append(0)
                else:
                    continue
        print('Data load finished')
        length = len(self.data)

        # 重新采样
        for i in range(length):
            contrib_a, contrib_v = contribution[i]
            # if 0.4 < contrib_a < 1:  # 0.5
            #add
            if 0.4 < contrib_a <= 1:  # 0.5
                for tt in range(1):
                    self.data.append(self.data[i])
                    self.label.append(self.label[i])
                    self.drop.append(2)
            elif -0.1 < contrib_a < 0.4:  # 0.0
                for tt in range(2):
                    self.data.append(self.data[i])
                    self.label.append(self.label[i])
                    self.drop.append(2)
            elif contrib_a < -0.1:  # -0.5
                for tt in range(3):
                    self.data.append(self.data[i])
                    self.label.append(self.label[i])
                    self.drop.append(2)

            # if 0.4 < contrib_v <1:  # 0.5
            #add
            if 0.4 < contrib_v <= 1:  # 0.5
                for tt in range(1):
                    self.data.append(self.data[i])
                    self.label.append(self.label[i])
                    self.drop.append(1)
            elif -0.1 < contrib_v < 0.4:
                for tt in range(2):
                    self.data.append(self.data[i])
                    self.label.append(self.label[i])
                    self.drop.append(1)
            elif contrib_v < -0.1:
                for tt in range(3):
                    self.data.append(self.data[i])
                    self.label.append(self.label[i])
                    self.drop.append(1)

        print('Data resample finished')

        self.mode = mode
        self.transforms = transforms

        self._init_atransform()

        print('# of files = %d ' % len(self.data))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        # 加载音频数据
        spectrogram = np.load(os.path.join(self.audio_path, av_file + '.npy'),allow_pickle=True)
        # spectrogram = pickle.load(open(self.data[idx], 'rb'))
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # 图像数据加载路径
        path = os.path.join(self.visual_path, av_file)  # 图像文件夹路径 #获取图像的数量
        image_files = [file for file in os.listdir(path) if file.endswith('.jpg')]  # 获取所有jpg文件


        # 如果图片数少于 3 张，则使用所有图片 add
        if len(image_files) < 3:
            selected_images = image_files  # 使用所有图片
        else:
            # 随机选择 3 张图片
            file_num=len(image_files)
            segment_size = file_num // 3
            segments = [
                (i * segment_size, (i + 1) * segment_size) for i in range(3)
            ]

            # 在每个段内随机选择一个索引
            selected_indices = [random.randint(segment[0], segment[1] - 1) for segment in segments]

            # 使用这些索引从图像文件列表中选取图片
            selected_images = [image_files[i] for i in selected_indices]
            # selected_images = random.sample(image_files, 3)

        # 图像数据增强
        if self.mode == 'train':
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # 图像数据增强
        image_arr = []
        for img_name in selected_images:
            img_path = os.path.join(path, img_name)
            # image = Image.open(img_path).convert('RGB')
            #add 转换RGB
            image = Image.open(img_path)
            image_arr.append(transf(image).unsqueeze(0).float())

        # 如果图像数量不足 3 张，补充空的图像（例如：全零图像） add
        if len(image_arr) < 3:
            # 使用一个全零的默认图像来填补
            # print("mask")
            default_image = torch.zeros_like(image_arr[0]) # 创建一个与原图像相同尺寸的零图像
            print ("zero")

            #掩码image【0】###########################
            # default_image = image_arr[0].clone()
            # default_image_np = default_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 假设张量在 CPU 上
            #
            # # 设置掩码参数
            # mask_prob = 1  # 掩码概率
            # mask_color = [0, 0, 0]
            # # 生成随机掩码并应用到图像上
            # height, width, _ = default_image_np.shape
            # random_mask = np.random.rand(height, width) < mask_prob
            # masked_image_np = default_image_np.copy()
            # masked_image_np[random_mask] = mask_color
            #
            # # 如果需要将掩码后的 NumPy 数组转换回 PyTorch 张量
            # # 注意：这里需要再次调整维度顺序，并可能需要将数据转换回 GPU（如果原始数据在 GPU 上）
            # masked_image_tensor = torch.from_numpy(masked_image_np.transpose(2, 0, 1)).float()
            # # 如果原始张量有批次维度，重新添加它
            # masked_image_tensor = masked_image_tensor.unsqueeze(0)
            # default_image = masked_image_tensor

            # ##############################


            # default_image = image_arr[0]
            while len(image_arr) < 3:
                image_arr.append(default_image)


        # 合并所有选择的图像
        image_n = torch.cat(image_arr, dim=0)

        label = self.label[idx]
        drop = self.drop[idx]



        return image_n, spectrogram, label, idx, drop


class AV_KS_Dataset_modality_level(Dataset):

    def __init__(self, mode, contribution_a, contribution_v, alpha, func='linear', transforms=None):
        self.data = []
        self.label = []
        self.drop = []

        # 根据模式选择路径
        if mode == 'train':
            csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/train.csv'  # 完整路径
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'
        elif mode == 'val':
            csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/val.csv'  # 完整路径
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'
        else:
            csv_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/test.csv'  # 完整路径
            self.audio_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Audio-01-FPS'
            self.visual_path = '/root/siton-data-zacharyData/VEMC/data/CEMA-D/Image-01-FPS'

            # 读取CSV文件并加载数据
        with open(csv_path) as f:
            for line in f:
                item = line.strip().split(",")  # 使用逗号分隔
                name = item[0]  # 提取文件名
                label = item[1]  # 提取标签

                # 检查音频文件是否存在
                audio_file = os.path.join(self.audio_path, name + '.npy')
                if os.path.exists(audio_file):
                    self.data.append(name)
                    self.label.append(LABEL_MAP[label])  # 将标签映射为数值
                    self.drop.append(0)

        print('data load finish')
        print(f"Data: {len(self.data)}, Labels: {len(self.label)},drop: {len(self.drop)}")
        length = len(self.data)

        # drop visual = 2, audio = 1, none = 0

        gap_a = 1.0 - contribution_a
        gap_v = 1.0 - contribution_v

        if func == 'linear':
            difference = (abs(gap_a - gap_v) / 3 * 2) * alpha
        elif func == 'tanh':
            tanh = torch.nn.Tanh()
            difference = tanh(torch.tensor((abs(gap_a - gap_v) / 3 * 2) * alpha))
        elif func == 'square':
            difference = (abs(gap_a - gap_v) / 3 * 2) ** 1.5 * alpha
        resample_num = int(difference * length)
        sample_choice = np.random.choice(length, resample_num)

        for i in sample_choice:
            self.data.append(self.data[i])
            self.label.append(self.label[i])
            if gap_a > gap_v:
                self.drop.append(2)
            else:
                self.drop.append(1)
        print('data resample finish')

        self.mode = mode
        self.transforms = transforms

        self._init_atransform()

        print('# of files = %d ' % len(self.data))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        # 加载音频数据
        spectrogram = np.load(os.path.join(self.audio_path, av_file + '.npy'))
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # 图像数据加载路径
        path = os.path.join(self.visual_path, av_file)  # 图像文件夹路径
        image_files = [file for file in os.listdir(path) if file.endswith('.jpg')]  # 获取所有jpg文件

        # 如果图片数少于 3 张，则使用所有图片
        if len(image_files) < 3:
            selected_images = image_files  # 使用所有图片
        else:
            # 随机选择 3 张图片
            file_num=len(image_files)
            segment_size = file_num // 3
            segments = [
                (i * segment_size, (i + 1) * segment_size) for i in range(3)
            ]

            # 在每个段内随机选择一个索引
            selected_indices = [random.randint(segment[0], segment[1] - 1) for segment in segments]

            # 使用这些索引从图像文件列表中选取图片
            selected_images = [image_files[i] for i in selected_indices]
            selected_images = random.sample(image_files, 3)

        # 图像数据增强
        if self.mode == 'train':
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # 图像数据增强
        image_arr = []
        for img_name in selected_images:
            img_path = os.path.join(path, img_name)
            image = Image.open(img_path).convert('RGB')
            image_arr.append(transf(image).unsqueeze(0).float())

        # 如果图像数量不足 3 张，补充空的图像（例如：全零图像）
        if len(image_arr) < 3:
            # 使用一个全零的默认图像来填补
            default_image = torch.zeros_like(image_arr[0])  # 创建一个与原图像相同尺寸的零图像
            while len(image_arr) < 3:
                image_arr.append(default_image)

        # 合并所有选择的图像
        image_n = torch.cat(image_arr, dim=0)

        label = self.label[idx]
        drop = self.drop[idx]

        return image_n, spectrogram, label, idx, drop