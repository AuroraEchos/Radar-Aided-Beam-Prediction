import os
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pro import RadarDataset
from model import RadarResNet3D

class Config:
    BATCH_SIZE = 8
    NUM_CLASSES = 64

    DATA_DIR = './Scenario9/development_dataset'
    TEST_CSV = os.path.join(DATA_DIR, 'scenario9_dev_test.csv')
    SAVE_DIR = './checkpoints'
    LOG_PATH = './test.log'

def evaluate_predictions(y, y_hat, ks=(1, 2, 3, 4, 5)):
    """
    返回每个 k 的 Top-k 百分比准确率和 beam distance
    """
    topk_accs = []
    max_k = max(ks)
    topk_pred = torch.topk(y_hat, k=max_k, dim=1).indices

    for k in ks:
        correct_k = (topk_pred[:, :k] == y.view(-1, 1)).any(dim=1).float().mean().item()
        topk_accs.append(correct_k * 100.0)

    # beam distance (取 Top-1)
    beam_dist = torch.mean(torch.abs(y.float() - topk_pred[:, 0].float()))

    return topk_accs, beam_dist

def test(model, device, dataloader, criterion, ks=(1, 2, 3, 4, 5)):
    """
    在测试集上评估
    """
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            all_outputs.append(outputs)
            all_labels.append(labels)

    # 拼接所有 batch
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    total_samples = all_labels.size(0)
    avg_loss = running_loss / total_samples

    # 计算指标
    topk_accs, beam_dist = evaluate_predictions(all_labels, all_outputs, ks=ks)

    return avg_loss, topk_accs, beam_dist

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_PATH),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_dataset = RadarDataset(Config.TEST_CSV, Config.DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    logger.info(f"Test samples: {len(test_dataset)}")

    # 初始化模型并加载权重
    model = RadarResNet3D(num_classes=Config.NUM_CLASSES).to(device)
    checkpoint_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()

    # 测试
    test_loss, topk_accs, beam_dist = test(model, device, test_loader, criterion)
    logger.info(f"Test Results -> loss={test_loss:.4f}, "
                f"Top-1={topk_accs[0]:.2f}%, Top-2={topk_accs[1]:.2f}%, Top-3={topk_accs[2]:.2f}%, Top-4={topk_accs[3]:.2f}%, Top-5={topk_accs[4]:.2f}%, "
                f"BeamDist={beam_dist:.4f}")

if __name__ == "__main__":
    main()
