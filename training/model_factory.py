import torch.nn as nn
from torchvision import models


def create_model(model_name, num_classes):

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    elif model_name == "densenet121":

        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )

    elif model_name == "mobilenet_v3_large":

        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes,
        )

    elif model_name == "vit_b_16":

        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT
        )

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes,
        )

    else:
        raise ValueError(f"модель не найдена...{model_name}")

    return model