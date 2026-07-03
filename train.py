from training.dataset_loader import get_loaders
from training.model_factory import create_model
from training.trainer import Trainer

from config import NUM_CLASSES


def train(model_name):

    # 1. данные
    train_loader, val_loader, class_names = get_loaders()

    print(f"\nClasses: {len(class_names)}")

    # 2. модель
    model = create_model(model_name, NUM_CLASSES)

    # 3. тренер
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        model_name=model_name
    )

    # 4. обучение
    trainer.train(epochs=10)

def train_all():

    models = [
        "resnet50",
        "densenet121",
        "efficientnet_b0",
        "mobilenet_v3_large",
        "vit_b_16"
    ]

    for model in models:

        print("\n" + "=" * 60)
        print(f"Запуск обучения модели: {model}")
        print("=" * 60)

        train(model)

if __name__ == "__main__":

    print("=" * 55)
    print("СИСТЕМА РАСПОЗНАВАНИЯ БЛЮД FOOD-101")
    print("=" * 55)

    print("\nВыберите режим обучения:")

    print("1 - Обучить одну модель")
    print("2 - Обучить все модели")

    choice = input("\nВаш выбор: ")

    if choice == "1":

        print("\nДоступные модели:")

        print("1 - ResNet50")
        print("2 - DenseNet121")
        print("3 - EfficientNet-B0")
        print("4 - MobileNetV3-Large")
        print("5 - ViT-B/16")

        model_choice = input("\nВведите номер модели: ")

        mapping = {
            "1": "resnet50",
            "2": "densenet121",
            "3": "efficientnet_b0",
            "4": "mobilenet_v3_large",
            "5": "vit_b_16"
        }

        train(mapping[model_choice])

    elif choice == "2":

        train_all()

    else:

        print("Неверный выбор.")