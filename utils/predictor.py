import torch
from PIL import Image
from torchvision import transforms

from config import DEVICE


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict(image_path, model, transform, class_names):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        values, indices = torch.topk(probs, k=3)

    results = []

    for v, i in zip(values[0], indices[0]):
        results.append({
            "class": class_names[i.item()],
            "probability": round(v.item() * 100, 2)
        })

    return results