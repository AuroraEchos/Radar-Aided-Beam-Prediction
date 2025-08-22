import os
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_pro import RadarDataset
from model import RadarResNet3D

class Config:
    BATCH_SIZE = 8
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 40
    NUM_CLASSES = 64

    DATA_DIR = './Scenario9/development_dataset'
    TRAIN_CSV = os.path.join(DATA_DIR, 'scenario9_dev_train.csv')
    VAL_CSV = os.path.join(DATA_DIR, 'scenario9_dev_val.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'scenario9_dev_test.csv')
    SAVE_DIR = './checkpoints'
    LOG_PATH = './train.log'

def evaluate_predictions(y, y_hat, ks=(1, 3, 5)):
    """
    返回每个 k 的 Top-k 百分比准确率和 beam distance
    """
    topk_accs = []
    max_k = max(ks)
    topk_pred = torch.topk(y_hat, k=max_k, dim=1).indices

    for k in ks:
        correct_k = (topk_pred[:, :k] == y.view(-1, 1)).any(dim=1).float().mean().item()
        topk_accs.append(correct_k * 100.0)

    # beam distance
    beam_dist = torch.mean(torch.abs(y.float() - topk_pred[:, 0].float()))

    return topk_accs, beam_dist

def train_epoch(model, device, dataloader, criterion, optimizer):
    """
    训练一个 epoch
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计 Top-1 准确率
        predictions = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (predictions == labels).sum().item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples * 100.0
    return epoch_loss, epoch_acc


def val_epoch(model, device, dataloader, criterion, ks=(1, 3, 5)):
    """
    验证一个 epoch
    返回: 平均 loss, Top-k 准确率列表, beam distance
    """
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            all_outputs.append(outputs)
            all_labels.append(labels)

    # 拼接所有 batch 的输出和标签
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    total_samples = all_labels.size(0)
    avg_loss = running_loss / total_samples

    # 计算 Top-k 准确率和 beam distance
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

    train_dataset = RadarDataset(Config.TRAIN_CSV, Config.DATA_DIR)
    val_dataset = RadarDataset(Config.VAL_CSV, Config.DATA_DIR)
    test_dataset = RadarDataset(Config.TEST_CSV, Config.DATA_DIR)
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    model = RadarResNet3D(num_classes=Config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    if not os.path.exists(Config.SAVE_DIR):
        os.makedirs(Config.SAVE_DIR)
    
    best_val_acc = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, topk_accs, beam_dist = val_epoch(model, device, val_loader, criterion)
        val_acc = topk_accs[0]  # Top-1

        logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} "
                    f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                    f"Val: loss={val_loss:.4f}, Top-1={val_acc:.2f}%, Top-3={topk_accs[1]:.2f}%, Top-5={topk_accs[2]:.2f}%, BeamDist={beam_dist:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model with Val Top-1 Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
