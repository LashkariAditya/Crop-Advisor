"""Validation-selected ImageNet transfer-learning experiment for soil photos.

The source dataset is non-commercial and has varying photo provenance. This
script is intended for local research/education, not commercial distribution.
The held-out test partition is evaluated once after selecting on validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]


def split_labels(labels: np.ndarray, seed: int = 7):
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == label))
        a, b = int(.7 * len(idx)), int(.85 * len(idx))
        for dst, chunk in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            dst.extend(chunk.tolist())
    return tuple(rng.permutation(p).tolist() for p in parts)


class SoilImages(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        with Image.open(path) as im:
            im = im.convert('RGB')
            return self.transform(im), label


def metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    support = cm.sum(axis=1)
    precision = np.divide(cm.diagonal(), cm.sum(axis=0), out=np.zeros(n_classes, float), where=cm.sum(axis=0) > 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(n_classes, float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes, float), where=precision + recall > 0)
    return {'accuracy': float(cm.trace() / max(1, cm.sum())), 'macro_f1': float(f1.mean()), 'per_class': [
        {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])}
        for i in range(n_classes)]}


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total, correct = 0., 0, 0
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward()
                optimizer.step()
        pred = logits.argmax(1)
        total_loss += float(loss.item()) * len(yb)
        total += len(yb)
        correct += int((pred == yb).sum())
        ys.extend(yb.cpu().numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
    return total_loss / max(1, total), correct / max(1, total), np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/raw/soil_classification_nc/Orignal-Dataset')
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--cyaug-data', default=None, help='optional CyAUG class folders; appended to original training partition only')
    ap.add_argument('--out', default='models/soil_transfer_latest.pt')
    args = ap.parse_args()
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = ROOT / args.data
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rows = []
    for name in classes:
        for path in sorted((root / name).rglob('*')):
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                try:
                    with Image.open(path) as im:
                        im.verify()
                    rows.append((path, class_to_idx[name]))
                except (OSError, Image.DecompressionBombError):
                    continue
    labels = np.array([y for _, y in rows])
    train_ids, val_ids, test_ids = split_labels(labels)
    weights = ResNet18_Weights.DEFAULT
    base = weights.transforms()
    image_size=args.image_size
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(.72, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20), transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.12),
        transforms.ToTensor(), transforms.Normalize(base.mean, base.std)])
    eval_tf = transforms.Compose([transforms.Resize(image_size+32), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize(base.mean, base.std)])
    train_rows = [rows[i] for i in train_ids]
    cyaug_added=0
    if args.cyaug_data:
        aug_root=ROOT/args.cyaug_data
        exact_originals=set()
        for original_path,_ in rows:
            exact_originals.add(hashlib.sha256(original_path.read_bytes()).digest())
        for name in classes:
            folder=aug_root/name
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob('*')):
                if path.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:
                    continue
                try:
                    if hashlib.sha256(path.read_bytes()).digest() in exact_originals:
                        continue
                    with Image.open(path) as im: im.verify()
                    train_rows.append((path,class_to_idx[name]))
                    cyaug_added+=1
                except (OSError,Image.DecompressionBombError):
                    continue
        print(f'Added {cyaug_added} CyAUG images to training only; validation/test remain original photos.',flush=True)
    loader = DataLoader(SoilImages(train_rows, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SoilImages([rows[i] for i in val_ids], eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(SoilImages([rows[i] for i in test_ids], eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    counts = np.bincount([label for _, label in train_rows], minlength=len(classes))
    class_weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Train the new head first, then fine-tune the last residual block.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    history, best_score, best_epoch, best_state, wait = [], (-1., -1.), 0, None, 0
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        if epoch == 3:
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {'params': model.layer4.parameters(), 'lr': 1e-5},
                {'params': model.fc.parameters(), 'lr': 2e-4}], weight_decay=1e-3)
        elif epoch == 0:
            optimizer = torch.optim.AdamW(model.fc.parameters(), lr=5e-4, weight_decay=1e-3)
        train_loss, train_acc, _, _ = run_epoch(model, loader, criterion, device, optimizer)
        val_loss, val_acc, vy, vp = run_epoch(model, val_loader, criterion, device)
        val_metrics = metrics(vy, vp, len(classes))
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_acc,
                        'validation_loss': val_loss, 'validation_accuracy': val_acc,
                        'validation_macro_f1': val_metrics['macro_f1']})
        score = (val_metrics['macro_f1'], val_acc)
        print(f"epoch {epoch+1:02d}/{args.epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch, wait = score, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 6:
                break
    # Only the validation-selected checkpoint is evaluated on test.
    model.load_state_dict(best_state)
    _, test_acc, ty, tp = run_epoch(model, test_loader, criterion, device)
    test_metrics = metrics(ty, tp, len(classes))
    torch.save({'state_dict': best_state, 'classes': classes, 'input_size': image_size,
                'architecture': 'torchvision.resnet18_imagenet_finetuned', 'seed': 7}, out)
    card = {'name': 'soil_photo_resnet18_transfer', 'task': 'soil', 'source_license': 'CC BY-NC 4.0 repository; underlying photo provenance may vary; research/education only pending rights review.',
            'trained_from_scratch': False, 'pretrained_weights': 'torchvision ResNet18 ImageNet1K_V1',
            'device': str(device), 'dataset': str(root), 'classes': classes,
            'additional_training_data': {'path': args.cyaug_data, 'images_added_after_exact_duplicate_removal': cyaug_added, 'type': 'CycleGAN-generated; source-photo lineage is not available for every image'} if args.cyaug_data else None,
            'image_size': image_size,
            'split': {'method': 'stratified random original-photo split, seed 7; validation/test never augmented', 'train_original': len(train_ids), 'train_with_synthetic': len(train_rows), 'validation': len(val_ids), 'test': len(test_ids)},
            'epochs_requested': args.epochs, 'epochs_trained': len(history), 'best_validation_epoch': best_epoch,
            'validation_selection': 'macro-F1, with accuracy as tie-breaker', 'validation_metrics': {k: v for k, v in history[best_epoch-1].items() if k.startswith('validation_')},
            'test_metrics': test_metrics, 'test_reused_in_prior_experiment': True,
            'history': history, 'warning': 'Small imbalanced dataset; random file split may contain related acquisition contexts across partitions. Test results are exploratory because this split was evaluated in prior experiments. Synthetic image lineage to the original training data is not fully known. Not field-validated.'}
    card_path=out.with_name(out.stem+'_card.json')
    card_path.write_text(json.dumps(card, indent=2), encoding='utf-8')
    print(json.dumps({'artifact': str(out), 'device': str(device), 'best_epoch': best_epoch,
                      'validation': card['validation_metrics'], 'test': test_metrics}, indent=2), flush=True)


if __name__ == '__main__':
    main()
