import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class RadarResNet3D(nn.Module):
    def __init__(self, num_classes=64):
        super().__init__()
        # 输入 2 通道 (实部 + 虚部)
        self.layer1 = ResidualBlock3D(2, 8, stride=1)
        self.layer2 = ResidualBlock3D(8, 16, stride=2)
        self.layer3 = ResidualBlock3D(16, 32, stride=2)
        self.layer4 = ResidualBlock3D(32, 64, stride=2)

        # 自适应池化到固定大小
        self.adapt_pool = nn.AdaptiveAvgPool3d((1, 4, 16))  # 输出 [B,64,1,4,16]

        # 全连接层
        self.fc1 = nn.Linear(64*1*4*16, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 2, D, H, W]
        x = self.layer1(x)   # [B,8,...]
        x = self.layer2(x)   # [B,16,...]
        x = self.layer3(x)   # [B,32,...]
        x = self.layer4(x)   # [B,64,...]
        x = self.adapt_pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # 测试模型
    model = RadarResNet3D(num_classes=64)
    x = torch.randn(8, 2, 16, 64, 128)  # [B, C, D, H, W]
    output = model(x)
    print(output.shape)  # 应该输出 [8, 64]