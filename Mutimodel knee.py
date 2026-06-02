import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, cast

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 6
    fusion_hidden: int = 256
    dropout: float = 0.3
    label_col: str = "klgrade"
    image_col: str = "filename"
    test_size: float = 0.2
    val_size: float = 0.1


CFG = Config()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_inverted_image(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)
    border = np.concatenate([
        arr[:10, :].ravel(),
        arr[-10:, :].ravel(),
        arr[:, :10].ravel(),
        arr[:, -10:].ravel()
    ])
    center = arr[arr.shape[0] // 4: 3 * arr.shape[0] // 4,
                 arr.shape[1] // 4: 3 * arr.shape[1] // 4].ravel()
    if border.mean() < center.mean():
        img = ImageOps.invert(img.convert("RGB"))
    return img


class MultimodalOADataset(Dataset):
    def __init__(self, df, image_root, image_col, tabular_cols, label_col, tabular_transform=None, train=False):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.image_col = image_col
        self.tabular_cols = tabular_cols
        self.label_col = label_col
        self.tabular_transform = tabular_transform
        self.train = train

        self.img_tf = transforms.Compose([
            transforms.Resize((CFG.image_size, CFG.image_size)),
            transforms.RandomHorizontalFlip(p=0.5 if train else 0.0),
            transforms.RandomRotation(10 if train else 0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def load_image(self, rel_path):
        path = self.image_root / str(rel_path)
        img = Image.open(path).convert("RGB")
        img = detect_inverted_image(img)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(row[self.image_col])
        x_img = self.img_tf(img)

        x_tab = row[self.tabular_cols].values
        if self.tabular_transform is not None:
            x_tab = self.tabular_transform.transform([x_tab])[0]
        x_tab = torch.tensor(x_tab, dtype=torch.float32)

        y = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return x_img, x_tab, y


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalClassifier(nn.Module):
    def __init__(self, tabular_dim, num_classes, fusion_hidden=256, dropout=0.3):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.image_backbone = efficientnet_v2_s(weights=weights)

        classifier = self.image_backbone.classifier
        classifier_block = classifier[1]
        img_feat_dim = int(cast(Any, classifier_block).in_features)
        self.image_backbone.classifier = nn.Sequential()

        self.tabular_encoder = TabularMLP(tabular_dim, hidden_dim=128, out_dim=128, dropout=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + 128, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_classes),
        )

    def forward(self, x_img, x_tab):
        img_feat = self.image_backbone(x_img)
        tab_feat = self.tabular_encoder(x_tab)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.fusion(fused)


def build_tabular_transform(train_df, tabular_cols):
    num_cols = [c for c in tabular_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in tabular_cols if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

    pre.fit(train_df[tabular_cols])
    return pre


def prepare_dataframe(df, label_col="klgrade"):
    df = df.copy()

    if label_col not in df.columns:
        candidates = [c for c in df.columns if "kl" in c.lower() and "grade" in c.lower()]
        if not candidates:
            raise ValueError(f"Label column '{label_col}' not found.")
        df = df.rename(columns={candidates[0]: label_col})

    return df


def stratified_split(df, label_col):
    train_df, temp_df = train_test_split(
        df, test_size=CFG.test_size, random_state=SEED, stratify=df[label_col]
    )
    val_rel = CFG.val_size / (1.0 - CFG.test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_rel, random_state=SEED, stratify=temp_df[label_col]
    )
    return train_df, val_df, test_df


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    all_y, all_p = [], []

    for x_img, x_tab, y in loader:
        x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x_img, x_tab)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_img, x_tab)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        all_y.extend(y.detach().cpu().numpy())
        all_p.extend(logits.argmax(1).detach().cpu().numpy())

    acc = accuracy_score(all_y, all_p)
    return total_loss / len(loader.dataset), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_y, all_p = [], []

    with torch.no_grad():
        for x_img, x_tab, y in loader:
            x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)
            logits = model(x_img, x_tab)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            all_y.extend(y.cpu().numpy())
            all_p.extend(logits.argmax(1).cpu().numpy())

    acc = accuracy_score(all_y, all_p)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y, all_p, average="macro", zero_division=0
    )
    return total_loss / len(loader.dataset), acc, prec, rec, f1, np.array(all_y), np.array(all_p)


def main(data_csv, image_root, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_csv)
    df = prepare_dataframe(df, CFG.label_col)

    if CFG.image_col not in df.columns:
        raise ValueError(f"Image column '{CFG.image_col}' not found.")

    tabular_cols = [c for c in df.columns if c not in [CFG.image_col, CFG.label_col]]
    if not tabular_cols:
        raise ValueError("No tabular columns found for clinical branch.")

    train_df, val_df, test_df = stratified_split(df, CFG.label_col)

    tab_pre = build_tabular_transform(train_df, tabular_cols)

    train_ds = MultimodalOADataset(train_df, image_root, CFG.image_col, tabular_cols, CFG.label_col, tab_pre, train=True)
    val_ds = MultimodalOADataset(val_df, image_root, CFG.image_col, tabular_cols, CFG.label_col, tab_pre, train=False)
    test_ds = MultimodalOADataset(test_df, image_root, CFG.image_col, tabular_cols, CFG.label_col, tab_pre, train=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

    sample_x_img, sample_x_tab, _ = train_ds[0]
    tab_dim = sample_x_tab.numel()
    num_classes = int(df[CFG.label_col].nunique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalClassifier(
        tabular_dim=tab_dim,
        num_classes=num_classes,
        fusion_hidden=CFG.fusion_hidden,
        dropout=CFG.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    class_counts = train_df[CFG.label_col].value_counts().sort_index()
    class_counts = class_counts.reindex(sorted(df[CFG.label_col].unique()), fill_value=1).values
    weights = torch.tensor(class_counts.sum() / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_f1 = -1
    patience = 0
    best_path = Path(output_dir) / "best_multimodal_oa.pt"
    history = []

    for epoch in range(1, CFG.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        va_loss, va_acc, va_prec, va_rec, va_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_f1)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_prec": va_prec,
            "val_rec": va_rec,
            "val_f1": va_f1,
        })

        if va_f1 > best_f1:
            best_f1 = va_f1
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "tab_preprocessor": tab_pre,
                "tabular_cols": tabular_cols,
                "label_col": CFG.label_col,
                "image_col": CFG.image_col,
                "num_classes": num_classes,
            }, best_path)
        else:
            patience += 1
            if patience >= CFG.patience:
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(Path(output_dir) / "training_history.csv", index=False)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(cm).to_csv(Path(output_dir) / "confusion_matrix.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(Path(output_dir) / "classification_report.csv")

    metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_prec_macro": float(test_prec),
        "test_rec_macro": float(test_rec),
        "test_f1_macro": float(test_f1),
    }
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    metrics = main(args.data_csv, args.image_root, args.output_dir)
    print(metrics)