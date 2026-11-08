"""Inference helpers for local NumPy regression artifacts."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import numpy as np


@lru_cache(maxsize=4)
def _image_model(path: str, modified_ns: int):
    """Keep image weights loaded between requests; key changes when artifact changes."""
    artifact = Path(path)
    if artifact.suffix == '.pt':
        import torch
        from torchvision import models
        checkpoint = torch.load(artifact, map_location='cpu', weights_only=True)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['classes']))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return (model, checkpoint['classes'], int(checkpoint.get('input_size', 224))), Lock()
    from ml.image.cnn import SmallCNN
    return SmallCNN.load(path), Lock()


def predict_image(path: Path, image: np.ndarray):
    """Predict from an RGB float image, reusing weights safely across requests."""
    model, lock = _image_model(str(path.resolve()), path.stat().st_mtime_ns)
    with lock:
        if path.suffix == '.pt':
            import torch
            network, classes, _ = model
            mean = torch.tensor([.485, .456, .406], dtype=torch.float32)[None, :, None, None]
            std = torch.tensor([.229, .224, .225], dtype=torch.float32)[None, :, None, None]
            tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(0, 3, 1, 2).float()
            tensor = (tensor - mean) / std
            with torch.inference_mode():
                views = torch.cat([tensor, tensor.flip(-1), tensor.flip(-2), tensor.flip((-2, -1))], dim=0)
                probabilities = torch.softmax(network(views), dim=1).mean(0).numpy()
            model_classes = classes
        else:
            probabilities = model.predict_proba(image)[0]
            model_classes = model.classes
    index = int(probabilities.argmax())
    return {"label": model_classes[index], "confidence": float(probabilities[index])}


def image_model_size(path: Path) -> int:
    """Return the cached model's expected square input dimension."""
    if path.suffix == '.pt':
        model, _ = _image_model(str(path.resolve()), path.stat().st_mtime_ns)
        return model[2]
    model, _ = _image_model(str(path.resolve()), path.stat().st_mtime_ns)
    return model.image_size


@lru_cache(maxsize=4)
def _crop_model(path: str, modified_ns: int):
    from ml.crop.model import MLPClassifier
    return MLPClassifier.load(path), Lock()


def predict_crop(path: Path, features: dict[str, float]):
    """Load a crop artifact once and serialize access to its mutable NumPy layers."""
    model, lock = _crop_model(str(path.resolve()), path.stat().st_mtime_ns)
    vector = np.array([[features[name] for name in model.features]], dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("Feature values must be finite.")
    with lock:
        return model.rank(vector)

def regression_prediction(path:Path,features:dict[str,float]):
    d=np.load(path,allow_pickle=False); names=d['features'].tolist()
    missing=[n for n in names if n not in features]
    if missing: raise ValueError('Missing trained feature values: '+', '.join(missing))
    x=np.array([[float(features[n]) for n in names]],dtype=float); x=(x-d['mean'])/(d['std']+1e-8)
    hidden=np.maximum(x@d['w1']+d['b1'],0); pred=(hidden@d['w2']+d['b2']).ravel()[0]*float(d['target_std'])+float(d['target_mean'])
    return {"prediction":float(pred),"features":names,"trained_from_scratch":True,"estimate":True}
