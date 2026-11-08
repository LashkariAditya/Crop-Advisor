"""Choose deterministic flip TTA on validation, then score test once."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights

from train_soil_transfer import ROOT, SoilImages, metrics, split_labels
from torchvision import transforms
from PIL import Image


def predict(model, loader, device, tta):
    ys, ps = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            if tta:
                views = torch.cat([x, x.flip(-1), x.flip(-2), x.flip((-2, -1))])
                logits = model(views).reshape(4, len(x), -1).mean(0)
            else:
                logits = model(x)
            ys.extend(y.numpy().tolist())
            ps.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.array(ys), np.array(ps)


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = ROOT / 'data/raw/soil_classification_nc/Orignal-Dataset'
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    idx = {c: i for i, c in enumerate(classes)}
    rows = [(p, idx[d.name]) for d in (root / c for c in classes) for p in sorted(d.rglob('*'))
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
    labels = np.array([y for _, y in rows])
    _, val_ids, test_ids = split_labels(labels)
    weights = ResNet18_Weights.DEFAULT
    base = weights.transforms()
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(base.mean, base.std)])
    val_loader = DataLoader(SoilImages([rows[i] for i in val_ids], tf), batch_size=16, shuffle=False)
    test_loader = DataLoader(SoilImages([rows[i] for i in test_ids], tf), batch_size=16, shuffle=False)
    checkpoint = torch.load(ROOT/'models/soil_transfer_latest.pt', map_location='cpu', weights_only=True)
    model = models.resnet18(weights=None)
    import torch.nn as nn
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint['classes']))
    model.load_state_dict(checkpoint['state_dict']); model.to(device)
    vy, vp = predict(model, val_loader, device, False)
    vy_tta, vp_tta = predict(model, val_loader, device, True)
    base_val, tta_val = metrics(vy, vp, len(classes)), metrics(vy_tta, vp_tta, len(classes))
    use_tta = (tta_val['macro_f1'], tta_val['accuracy']) > (base_val['macro_f1'], base_val['accuracy'])
    ty, tp = predict(model, test_loader, device, use_tta)
    result = {'selection': 'validation macro-F1, accuracy tie-breaker', 'tta_selected': use_tta,
              'validation_without_tta': base_val, 'validation_with_flip_tta': tta_val,
              'test_selected_method': metrics(ty, tp, len(classes)),
              'note': 'The test split was previously scored for the no-TTA checkpoint; this second comparison is exploratory.'}
    (ROOT/'models/soil_transfer_tta_evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
