# ============================================================
# Chest X-Ray Pneumonia - Training
# ============================================================

import gc
import math
import time
from pathlib import Path

import torch

from ailib.device import get_device
from ailib.experiment import (
    EpochResult,
    ExperimentConfig,
    ExperimentResult,
    save_checkpoint,
    load_checkpoint,
    save_epoch_history,
)
from ailib.resource import check_resource_safety


# ============================================================
# Device
# ============================================================

device = get_device()


# ============================================================
# Utility
# ============================================================

def is_finite(value):
    """
    Check whether a numeric value is finite.
    """
    return math.isfinite(float(value))


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp=False,
):
    """
    Train the model for one epoch.

    Returns
    -------
    loss : float
    accuracy : float
    stop_reason : str | None
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.perf_counter()

    loader_iter = iter(loader)

    for batch_idx in range(len(loader)):

        # ----------------------------------------------------
        # Data Load
        # ----------------------------------------------------

        load_start = time.perf_counter()

        images, labels = next(loader_iter)

        load_time = (
            time.perf_counter()
            - load_start
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        model_start = time.perf_counter()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # ----------------------------------------------------
        # Non-finite Loss Check
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            print(
                f"\n[Train Batch {batch_idx + 1}] "
                f"Non-finite loss detected: {loss.item()}"
            )

            return (
                float("nan"),
                float("nan"),
                "non_finite_train_loss",
            )

        # ----------------------------------------------------
        # Backward / Optimizer
        # ----------------------------------------------------

        if use_amp:

            scaler.scale(loss).backward()

            gradients_finite = True

            for parameter in model.parameters():

                if parameter.grad is not None:

                    if not torch.isfinite(
                        parameter.grad
                    ).all():

                        gradients_finite = False
                        break

            if not gradients_finite:

                print(
                    f"\n[Train Batch {batch_idx + 1}] "
                    f"Non-finite gradient detected."
                )

                return (
                    float("nan"),
                    float("nan"),
                    "non_finite_train_gradient",
                )

            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()

            gradients_finite = True

            for parameter in model.parameters():

                if parameter.grad is not None:

                    if not torch.isfinite(
                        parameter.grad
                    ).all():

                        gradients_finite = False
                        break

            if not gradients_finite:

                print(
                    f"\n[Train Batch {batch_idx + 1}] "
                    f"Non-finite gradient detected."
                )

                return (
                    float("nan"),
                    float("nan"),
                    "non_finite_train_gradient",
                )

            optimizer.step()

        # ----------------------------------------------------
        # CUDA Synchronization
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        model_time = (
            time.perf_counter()
            - model_start
        )

        # ----------------------------------------------------
        # Resource
        # ----------------------------------------------------

        if device.type == "cuda":

            gpu_allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            gpu_reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

        else:

            gpu_allocated = 0.0
            gpu_reserved = 0.0

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        if (
            batch_idx == 0
            or (batch_idx + 1) % 50 == 0
        ):

            print(
                f"[Batch {batch_idx + 1}/{len(loader)}] "
                f"Data Load: {load_time:.3f}s | "
                f"Model: {model_time:.3f}s | "
                f"Total: "
                f"{load_time + model_time:.3f}s | "
                f"GPU Alloc: "
                f"{gpu_allocated:.2f}GB | "
                f"GPU Reserved: "
                f"{gpu_reserved:.2f}GB"
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_loss = loss.item()

        total_loss += (
            batch_loss
            * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    # --------------------------------------------------------
    # Epoch Result
    # --------------------------------------------------------

    if total == 0:

        return (
            float("nan"),
            float("nan"),
            "empty_train_loader",
        )

    loss = total_loss / total
    accuracy = correct / total

    # --------------------------------------------------------
    # Final Finite Check
    # --------------------------------------------------------

    if not is_finite(loss):

        return (
            float("nan"),
            float("nan"),
            "non_finite_train_loss",
        )

    if not is_finite(accuracy):

        return (
            loss,
            float("nan"),
            "non_finite_train_accuracy",
        )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print(
        f"[Train Epoch] "
        f"Time: {epoch_time:.2f}s | "
        f"Loss: {loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )

    return (
        loss,
        accuracy,
        None,
    )


# ============================================================
# Validation One Epoch
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    use_amp=False,
):
    """
    Validate the model for one epoch.

    Returns
    -------
    loss : float
    accuracy : float
    stop_reason : str | None
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.perf_counter()

    loader_iter = iter(loader)

    for batch_idx in range(len(loader)):

        # ----------------------------------------------------
        # Data Load
        # ----------------------------------------------------

        load_start = time.perf_counter()

        images, labels = next(loader_iter)

        load_time = (
            time.perf_counter()
            - load_start
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        model_start = time.perf_counter()

        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # ----------------------------------------------------
        # Non-finite Loss Check
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            print(
                f"\n[Validation Batch {batch_idx + 1}] "
                f"Non-finite loss detected: {loss.item()}"
            )

            return (
                float("nan"),
                float("nan"),
                "non_finite_validation_loss",
            )

        # ----------------------------------------------------
        # CUDA Synchronization
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        model_time = (
            time.perf_counter()
            - model_start
        )

        # ----------------------------------------------------
        # Resource
        # ----------------------------------------------------

        if device.type == "cuda":

            gpu_allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            gpu_reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

        else:

            gpu_allocated = 0.0
            gpu_reserved = 0.0

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        if (
            batch_idx == 0
            or (batch_idx + 1) % 10 == 0
        ):

            print(
                f"[Val Batch {batch_idx + 1}/{len(loader)}] "
                f"Data Load: {load_time:.3f}s | "
                f"Model: {model_time:.3f}s | "
                f"Total: "
                f"{load_time + model_time:.3f}s | "
                f"GPU Alloc: "
                f"{gpu_allocated:.2f}GB | "
                f"GPU Reserved: "
                f"{gpu_reserved:.2f}GB"
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_loss = loss.item()

        total_loss += (
            batch_loss
            * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    # --------------------------------------------------------
    # Epoch Result
    # --------------------------------------------------------

    if total == 0:

        return (
            float("nan"),
            float("nan"),
            "empty_validation_loader",
        )

    loss = total_loss / total
    accuracy = correct / total

    # --------------------------------------------------------
    # Final Finite Check
    # --------------------------------------------------------

    if not is_finite(loss):

        return (
            float("nan"),
            float("nan"),
            "non_finite_validation_loss",
        )

    if not is_finite(accuracy):

        return (
            loss,
            float("nan"),
            "non_finite_validation_accuracy",
        )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print(
        f"[Validation Epoch] "
        f"Time: {epoch_time:.2f}s | "
        f"Loss: {loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )

    return (
        loss,
        accuracy,
        None,
    )


# ============================================================
# Experiment Training
# ============================================================

def run_experiment(
    experiment_name,
    optimizer_name,
    num_epochs,
    train_loader,
    val_loader=None,
    learning_rate=0.001,
    weight_decay=0.0,
    model=None,
    image_size=512,
    channels=3,
    normalization="imagenet_mean_std",
    augmentation="none",
    loss_name="CrossEntropy",
    checkpoint_dir=None,
):
    """
    Run a training experiment.

    Parameters
    ----------
    num_epochs : int
        Target total number of epochs.

        If a checkpoint exists, training resumes from the
        last completed epoch until num_epochs.

    checkpoint_dir : str | Path | None
        Directory where checkpoint.pth is stored.

        If None, checkpoint recovery is disabled.
    """

    # ========================================================
    # Device / AMP
    # ========================================================

    # AMP is intentionally disabled for the current project.
    use_amp = False

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print(
        f"Device: {device}"
    )

    print(
        f"AMP: {'ON' if use_amp else 'OFF'}"
    )

    # ========================================================
    # Model
    # ========================================================

    if model is None:

        raise ValueError(
            "model must be provided."
        )

    model = model.to(device)

    # ========================================================
    # Optimizer
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:

        raise ValueError(
            "No trainable parameters found."
        )

    if optimizer_name == "Adam":

        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "SGD":

        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: "
            f"{optimizer_name}"
        )

    # ========================================================
    # Loss
    # ========================================================

    if loss_name == "CrossEntropy":

        criterion = torch.nn.CrossEntropyLoss()

    else:

        raise ValueError(
            f"Unsupported loss: {loss_name}"
        )

    # ========================================================
    # Experiment Config
    # ========================================================

    config = ExperimentConfig(
        experiment_name=experiment_name,
        model=model.__class__.__name__,
        image_size=image_size,
        channels=channels,
        normalization=normalization,
        augmentation=augmentation,
        batch_size=train_loader.batch_size,
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        use_amp=use_amp,
        loss=loss_name,
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint_path = None

    if checkpoint_dir is not None:

        checkpoint_path = (
            Path(checkpoint_dir)
            / "checkpoint.pth"
        )

    if checkpoint_path is not None:

        (
            start_epoch,
            best_epoch,
            best_validation_loss,
            best_validation_accuracy,
            history,
        ) = load_checkpoint(
            filepath=checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    else:

        start_epoch = 0

        history = []

        best_epoch = None
        best_validation_loss = None
        best_validation_accuracy = None

    # ========================================================
    # Target Epoch Check
    # ========================================================

    if start_epoch >= num_epochs:

        print(
            f"\n[{experiment_name}] "
            f"Checkpoint already reached target "
            f"epochs: {num_epochs}"
        )

    else:

        print(
            f"\n[{experiment_name}] "
            f"Training range: "
            f"Epoch {start_epoch + 1} "
            f"-> {num_epochs}"
        )

    # ========================================================
    # Training Status
    # ========================================================

    training_stopped = False
    stop_reason = None

    experiment_start = time.perf_counter()

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        num_epochs,
    ):

        epoch_start = time.perf_counter()

        # ----------------------------------------------------
        # Resource Safety Check
        # ----------------------------------------------------

        safe, reason = check_resource_safety(
            device
        )

        if not safe:

            training_stopped = True
            stop_reason = reason

            print(
                f"\n[{experiment_name}] "
                f"Training stopped: "
                f"{stop_reason}"
            )

            break

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        try:

            (
                train_loss,
                train_accuracy,
                train_stop_reason,
            ) = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
            )

            # ------------------------------------------------
            # Train Non-finite Check
            # ------------------------------------------------

            if train_stop_reason is not None:

                training_stopped = True
                stop_reason = train_stop_reason

                print(
                    f"\n[{experiment_name}] "
                    f"Training stopped: "
                    f"{stop_reason}"
                )

                # Do NOT save checkpoint.
                # The previous completed epoch remains
                # the recovery point.

                break

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_loss = None
            val_accuracy = None

            if val_loader is not None:

                (
                    val_loss,
                    val_accuracy,
                    val_stop_reason,
                ) = validate_one_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    use_amp=use_amp,
                )

                # --------------------------------------------
                # Validation Non-finite Check
                # --------------------------------------------

                if val_stop_reason is not None:

                    training_stopped = True
                    stop_reason = val_stop_reason

                    print(
                        f"\n[{experiment_name}] "
                        f"Training stopped: "
                        f"{stop_reason}"
                    )

                    # Do NOT save checkpoint.
                    # The previous completed epoch remains
                    # the recovery point.

                    break

        except torch.cuda.OutOfMemoryError:

            training_stopped = True
            stop_reason = (
                "CUDA Out Of Memory"
            )

            print(
                f"\n[{experiment_name}] "
                f"Training stopped: "
                f"{stop_reason}"
            )

            # Do NOT save checkpoint.
            # The previous completed epoch remains
            # the recovery point.

            break

        # ----------------------------------------------------
        # Epoch Result
        # ----------------------------------------------------

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        epoch_result = EpochResult(
            experiment_name=experiment_name,
            epoch=epoch + 1,
            num_epochs=num_epochs,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            epoch_time=epoch_time,
            validation_loss=val_loss,
            validation_accuracy=val_accuracy,
        )

        save_epoch_history(
            history,
            epoch_result,
        )

        # ----------------------------------------------------
        # Best Validation Result
        # ----------------------------------------------------

        if (
            val_loss is not None
            and val_accuracy is not None
            and is_finite(val_loss)
            and is_finite(val_accuracy)
        ):

            if (
                best_validation_loss is None
                or val_loss < best_validation_loss
            ):

                best_validation_loss = val_loss

                best_validation_accuracy = (
                    val_accuracy
                )

                best_epoch = epoch + 1

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if checkpoint_path is not None:

            save_checkpoint(
                filepath=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_epoch=best_epoch,
                best_validation_loss=(
                    best_validation_loss
                ),
                best_validation_accuracy=(
                    best_validation_accuracy
                ),
                history=history,
            )

        # ----------------------------------------------------
        # Epoch Output
        # ----------------------------------------------------

        if val_loss is not None:

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_accuracy:.4f} "
                f"Time: {epoch_time:.2f}s"
            )

        else:

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"Time: {epoch_time:.2f}s"
            )

    # ========================================================
    # Training Time
    # ========================================================

    training_time = (
        time.perf_counter()
        - experiment_start
    )

    # ========================================================
    # Experiment Result
    # ========================================================

    experiment_result = ExperimentResult(
        experiment_name=experiment_name,
        training_time=training_time,
        stopped=training_stopped,
        stop_reason=stop_reason,
        best_epoch=best_epoch,
        best_validation_loss=(
            best_validation_loss
        ),
        best_validation_accuracy=(
            best_validation_accuracy
        ),
    )

    # ========================================================
    # Memory Cleanup
    # ========================================================

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return (
        model,
        config,
        history,
        experiment_result,
    )