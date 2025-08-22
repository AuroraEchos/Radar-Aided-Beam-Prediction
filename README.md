### Residual 3D CNN for Radar Beam Prediction

基于 **Residual 3D CNN** 的毫米波雷达波束预测方法。该项目利用 **DeepSense 6G** 数据集中的雷达回波，构建 **Range-Angle-Doppler (RAD) Cube**，并通过 **复数输入 + 残差 3D 卷积网络** 进行波束预测。在 **Top-5 精度** 和 **Beam Distance** 指标上取得一定的效果。

#### 🔥 特点

- **复数输入 (Real + Imag)**：保留幅度与相位信息，提高预测精度
- **Residual 3D CNN**：高效建模时域、空间和频域特征
- **端到端训练**：无需显式信道估计，直接从雷达数据预测波束
- **实验效果**：在 DeepSense 6G 数据集上取得 **90.39% Top-5 准确率**

#### 📂 项目结构

```
.
├── data_pro.py       # 数据预处理与Dataset定义
├── model.py          # Residual 3D CNN 模型实现
├── train.py          # 训练入口
├── test.py           # 模型评估
└── README.md         # 项目说明
```

#### 📊 数据集

本项目基于 **DeepSense 6G Dataset**。请从 

[官方页面]: https://www.deepsense6g.net/radar-aided-beam-prediction/

下载相关场景数据并放在根目录下。

#### 🚀 训练

```
python train.py
```

#### 📈 评估

```
python test.py
```

**在 DeepSense 6G 场景 9 上的实验结果：**

| Top-1  | Top-2 | Top-3 | Top-4 | Top-5 | BeamDist |
| ------ | ----- | ----- | ----- | ----- | -------- |
| 41.48% | 60.37 | 75.04 | 85.50 | 90.39 | 1.2985   |