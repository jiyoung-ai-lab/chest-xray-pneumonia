# ============================================================
# Chest X-Ray Pneumonia - Evaluation
# ============================================================

import torch

from ailib.transform_v2 import create_transform
from ailib.dataset_v2 import ImageDataset
from ailib.dataloader_v2 import create_dataloader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# Evaluation DataLoader
# ============================================================

def create_eval_loader(
    dataframe,
    data_dir,
    class_to_idx,
    image_size,
    mean,
    std,
    batch_size,
):
    """
    Create DataLoader for validation or test evaluation.
    """

    eval_transform = create_transform(
        image_size=image_size,
        mean=mean,
        std=std,
    )

    eval_dataset = ImageDataset(
        dataframe=dataframe,
        data_dir=data_dir,
        class_to_idx=class_to_idx,
        transform=eval_transform,
    )

    eval_loader = create_dataloader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return eval_loader


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device,
):
    """
    Generate labels, predictions, and positive-class probabilities.
    """

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

    return (
        all_labels,
        all_predictions,
        all_probabilities,
    )


# ============================================================
# Classification Metrics
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities,
):
    """
    Calculate classification metrics.
    """

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(
    model,
    loader,
    device,
):
    """
    Run prediction and calculate classification metrics.
    """

    labels, predictions, probabilities = predict(
        model=model,
        loader=loader,
        device=device,
    )

    metrics = calculate_metrics(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )

    return metrics
 

import json
from pathlib import Path


def save_validation_result(
    run_path,
    experiment,
    run,
    labels,
    predictions,
    probabilities,
    metrics,
):
    validation_path = Path(run_path) / "validation.json"

    result = {
        "experiment": experiment,
        "run": run,
        "dataset": "validation",
        "num_samples": len(labels),
        "metrics": metrics,
        "predictions": {
            "labels": labels.tolist(),
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        },
    }

    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return validation_path    

def save_test_result(
    run_path,
    experiment,
    run,
    labels,
    predictions,
    probabilities,
    metrics,
):
    test_path = Path(run_path) / "test.json"

    result = {
        "experiment": experiment,
        "run": run,
        "dataset": "test",
        "num_samples": len(labels),
        "metrics": metrics,
        "predictions": {
            "labels": labels.tolist(),
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        },
    }

    with open(
        test_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return test_path