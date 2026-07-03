from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from config import DATASET_DIR, BATCH_SIZE


class Food101Dataset(Dataset):

    def __init__(self, txt_file, images_dir, transform=None):

        self.samples = []
        self.transform = transform
        self.images_dir = images_dir

        with open(txt_file, "r") as f:
            for line in f:
                path = line.strip()
                label = path.split("/")[0]
                self.samples.append((path, label))

        self.classes = sorted(list(set([s[1] for s in self.samples])))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img_path = os.path.join(self.images_dir, path + ".jpg")
        image = Image.open(img_path).convert("RGB")

        label = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_loaders():

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    images_dir = DATASET_DIR / "images"

    train_dataset = Food101Dataset(
        DATASET_DIR / "meta" / "train.txt",
        images_dir,
        transform=train_tf
    )

    val_dataset = Food101Dataset(
        DATASET_DIR / "meta" / "test.txt",
        images_dir,
        transform=val_tf
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes

def load_class_names():
    from config import DATASET_DIR

    classes = []

    with open(DATASET_DIR / "meta" / "classes.txt", "r") as f:
        for line in f:
            classes.append(line.strip())

    return classes

def get_transforms(model_name=None):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])