import torch
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scipy.io as sio

def RAD_Cube_complex(data, fft_angle=64, fft_doppler=16):
    """
    返回复数通道: [2, angle_bins, range_bins, doppler_bins]
    """
    # 去均值
    data = data - np.mean(data)

    # Range FFT
    data = np.fft.fft(data, axis=1)

    # Doppler FFT
    data = np.fft.fft(data, n=fft_doppler, axis=2)

    # Angle FFT
    data = np.fft.fft(data, n=fft_angle, axis=0)

    # 提取实部和虚部
    real = np.real(data)
    imag = np.imag(data)

    # 归一化
    mean_r, std_r = np.mean(real), np.std(real)
    mean_i, std_i = np.mean(imag), np.std(imag)
    real = (real - mean_r) / (std_r + 1e-6)
    imag = (imag - mean_i) / (std_i + 1e-6)

    rad_complex = np.stack([real, imag], axis=0)  # [2, angle, range, doppler]
    return rad_complex.astype(np.float32)



class RadarDataset(Dataset):
    def __init__(self, csv_file, root_dir, fft_angle=64, fft_doppler=16, to_tensor=True):
        """
        Args:
            csv_file (str): CSV 文件路径
            root_dir (str): 雷达数据根目录
            fft_angle (int): Angle FFT 零填充长度
            fft_doppler (int): Doppler FFT 压缩长度
            to_tensor (bool): 是否将输出转换为 torch.Tensor
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.fft_angle = fft_angle
        self.fft_doppler = fft_doppler
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        data_path = os.path.join(self.root_dir, row["unit1_radar_1"])
        
        # 加载原始雷达数据
        radar_mat = sio.loadmat(data_path)
        key = list(radar_mat.keys())[-1]  # 默认取最后一个 key
        radar_data = radar_mat[key]       # shape: [num_rx, num_samples_per_chirp, num_chirps_per_frame]
        
        rad_cube = RAD_Cube_complex(radar_data, fft_angle=self.fft_angle, fft_doppler=self.fft_doppler)
        rad_cube = rad_cube.transpose(0, 3, 1, 2)  # [2, doppler_bins, angle_bins, range_bins]


        if self.to_tensor:
            rad_cube = torch.from_numpy(rad_cube)  # 转为 torch.Tensor

        # 标签
        label = int(row["beam_index_1"]) - 1
        label = torch.tensor(label, dtype=torch.long)

        return rad_cube, label


if __name__ == "__main__":
    csv_file = "Scenario9/development_dataset/scenario9_dev_train.csv"
    root_dir = "Scenario9/development_dataset/"
    dataset = RadarDataset(csv_file, root_dir)

    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}")  # [doppler_bins, angle_bins, range_bins]
    print(f"Label: {label}")
