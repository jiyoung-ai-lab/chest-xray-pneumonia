
# ============================================================
# Chest X-Ray Pneumonia - Model
# ============================================================

import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# Simple CNN
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# Transfer Learning
# ============================================================

def build_transfer_model(
    model_name="resnet50",
    num_classes=2,
    in_channels=3,
    pretrained=True,
):

    # ========================================================
    # ResNet50
    # ========================================================

    if model_name == "resnet50":

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet50(
            weights=weights
        )

        # ----------------------------------------------------
        # ResNet50의 원래 conv1을 그대로 사용
        # RGB 3채널 입력
        # ----------------------------------------------------

        if in_channels != 3:
            raise ValueError(
                "ResNet50 pretrained RGB test requires "
                "in_channels=3."
            )

        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    # ========================================================
    # DenseNet121
    # ========================================================

    elif model_name == "densenet121":

        weights = (
            models.DenseNet121_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.densenet121(
            weights=weights
        )

        old_conv = model.features.conv0

        if in_channels != 3:

            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            if pretrained:

                with torch.no_grad():
                    new_conv.weight.copy_(
                        old_conv.weight.mean(
                            dim=1,
                            keepdim=True,
                        )
                    )

            model.features.conv0 = new_conv

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )

    # ========================================================
    # EfficientNet-B0
    # ========================================================

    elif model_name == "efficientnet_b0":

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.efficientnet_b0(
            weights=weights
        )

        old_conv = model.features[0][0]

        if in_channels != 3:

            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            if pretrained:

                with torch.no_grad():
                    new_conv.weight.copy_(
                        old_conv.weight.mean(
                            dim=1,
                            keepdim=True,
                        )
                    )

            model.features[0][0] = new_conv

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return model


# ============================================================
# Fine-Tuning Configuration
# ============================================================

def configure_fine_tuning(
    model,
    model_name,
    fine_tuning_mode,
):

    for parameter in model.parameters():
        parameter.requires_grad = False

    if fine_tuning_mode == "frozen":

        if model_name == "resnet50":

            for parameter in model.fc.parameters():
                parameter.requires_grad = True

        elif model_name == "densenet121":

            for parameter in model.classifier.parameters():
                parameter.requires_grad = True

        elif model_name == "efficientnet_b0":

            for parameter in model.classifier.parameters():
                parameter.requires_grad = True

    elif fine_tuning_mode == "partial":

        if model_name == "resnet50":

            for parameter in model.layer4.parameters():
                parameter.requires_grad = True

            for parameter in model.fc.parameters():
                parameter.requires_grad = True

        elif model_name == "densenet121":

            for parameter in model.features.denseblock4.parameters():
                parameter.requires_grad = True

            for parameter in model.features.norm5.parameters():
                parameter.requires_grad = True

            for parameter in model.classifier.parameters():
                parameter.requires_grad = True

        elif model_name == "efficientnet_b0":

            for parameter in model.features[-1].parameters():
                parameter.requires_grad = True

            for parameter in model.classifier.parameters():
                parameter.requires_grad = True

    elif fine_tuning_mode == "full":

        for parameter in model.parameters():
            parameter.requires_grad = True

    else:

        raise ValueError(
            f"Unsupported fine-tuning mode: "
            f"{fine_tuning_mode}"
        )

    return model


# ============================================================
# Trainable Parameter Check
# ============================================================

def check_trainable_parameters(model):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    trainable_ratio = (
        trainable_parameters
        / total_parameters
        if total_parameters > 0
        else 0.0
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "trainable_ratio": trainable_ratio,
    }
 
