import os
import glob
import random
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Dataset
class GapDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, input_size: Tuple[int, int] = (256, 256), augment: bool = True):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in self.img_paths]
        self.input_size = input_size
        self.augment = augment

        # keep only pairs that exist
        pairs = []
        for ip, mp in zip(self.img_paths, self.mask_paths):
            if os.path.exists(mp):
                pairs.append((ip, mp))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, mask):
        # Horizontal flip
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Brightness/contrast jitter (light)
        if random.random() < 0.7:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-20, 20)    # brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Gaussian blur occasionally
        if random.random() < 0.2:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to read {img_path} or {mask_path}")

        # Resize
        H, W = self.input_size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # Convert mask to 0/1
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        # Normalize image to [0,1]
        img = img.astype(np.float32) / 255.0
        # CHW for torch
        img = np.transpose(img, (2, 0, 1))
        mask = mask[None, :, :]  # 1xHxW

        return torch.tensor(img), torch.tensor(mask)
    


#Build simple U net
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
    
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        m = self.mid(p4)

        u4 = self.up4(m)
        c4 = self.conv4(torch.cat([u4, d4], dim=1))

        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(c1)


#Loss: BCE+ Dice:
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

#train
def train():
    random.seed(42)
    torch.manual_seed(42)

    img_dir = "ml_dataset/images"
    mask_dir = "ml_dataset/masks"
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    input_size = (256, 256)
    batch_size = 8
    epochs = 10
    lr = 1e-3

    dataset = GapDataset(img_dir, mask_dir, input_size=input_size, augment=True)
    if len(dataset) < 10:
        raise RuntimeError(f"Not enough training samples found: {len(dataset)}. Generate ml_dataset first.")

    # simple split
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, out_ch=1, base=32).to(device)

    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = 1e9
    best_path = os.path.join(out_dir, "gap_unet_best.pt")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        tr_loss /= max(1, len(train_loader))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce(logits, y) + dice_loss(logits, y)
                va_loss += loss.item()
        va_loss /= max(1, len(val_loader))

        print(f"Epoch {ep}/{epochs} train={tr_loss:.4f} val={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

    # Export ONNX
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    onnx_path = os.path.join(out_dir, "gap_unet.onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo = False,
    )

    print(f"Saved ONNX model: {onnx_path}")


if __name__ == "__main__":
    train()