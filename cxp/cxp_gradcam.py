# ============================================================
# cxp_gradcam.py
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:
    """
    Grad-CAM for CNN-based classification models.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output

        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, image, target_class):
        """
        Generate Grad-CAM heatmap.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor with shape (1, C, H, W).

        target_class : int
            Target class index.

        Returns
        -------
        torch.Tensor
            Normalized heatmap with shape (1, H, W).
        """

        self.model.eval()
        self.activations = None
        self.gradients = None

        image = image.detach().requires_grad_(True)

        output = self.model(image)

        if output.ndim != 2:
            raise RuntimeError(
                f"Expected classification output with shape "
                f"(batch, classes), got {output.shape}."
            )

        score = output[:, target_class].sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        gradients = self.gradients
        activations = self.activations

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )

        if activations is None:
            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )

        if activations.ndim != 4:
            raise RuntimeError(
                f"Expected CNN feature map with 4 dimensions, "
                f"got {activations.shape}."
            )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cam_min = cam.amin(
            dim=(1, 2),
            keepdim=True,
        )

        cam_max = cam.amax(
            dim=(1, 2),
            keepdim=True,
        )

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return cam.detach()


def get_gradcam_target_layer(model, model_name):
    """
    Get a suitable Grad-CAM target layer for a CNN model.
    """

    model_name = model_name.lower()

    if model_name == "resnet":
        return model.layer4[-1]

    if model_name in {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    }:
        return model.layer4[-1]

    if model_name in {
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "densenet",
    }:
        return model.features.denseblock4

    raise ValueError(
        f"Unsupported Grad-CAM model: {model_name}"
    )

# ============================================================
# Image
# ============================================================

def load_gradcam_image(image_path):
    """
    Load an image for Grad-CAM visualization.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return Image.open(image_path).convert("RGB")


def prepare_gradcam_image(image, transform, device):
    """
    Convert PIL image to model input tensor.
    """

    image_tensor = transform(image)

    return image_tensor.unsqueeze(0).to(device)


# ============================================================
# Single Sample
# ============================================================

def generate_gradcam_sample(
    gradcam,
    image_path,
    transform,
    target_class,
    device,
):
    """
    Generate Grad-CAM for a single image.

    Returns
    -------
    image : PIL.Image.Image
        Original image.

    heatmap : numpy.ndarray
        Grad-CAM heatmap.

    """

    image = load_gradcam_image(image_path)

    image_tensor = prepare_gradcam_image(
        image=image,
        transform=transform,
        device=device,
    )

    heatmap = gradcam.generate(
        image=image_tensor,
        target_class=target_class,
    )

    heatmap = (
        heatmap.squeeze()
        .cpu()
        .numpy()
    )

    return image, heatmap


# ============================================================
# Visualization
# ============================================================

def plot_gradcam_samples(
    dataframe,
    gradcam,
    transform,
    image_dir,
    n_samples=4,
    title_prefix="Grad-CAM",
    device=None,
):
    """
    Plot original images and Grad-CAM overlays.

    The dataframe must contain:
        Folder
        File
        y_pred
        y_prob
    """

    if dataframe.empty:
        print("No samples available for Grad-CAM.")
        return

    if device is None:
        device = next(gradcam.model.parameters()).device

    samples = dataframe.head(n_samples)

    n_samples = len(samples)

    fig, axes = plt.subplots(
        2,
        n_samples,
        figsize=(4 * n_samples, 8),
        squeeze=False,
    )

    for col, (_, sample) in enumerate(
        samples.iterrows()
    ):

        image_path = (
            Path(image_dir)
            / sample["Folder"]
            / sample["File"]
        )

        image, heatmap = generate_gradcam_sample(
            gradcam=gradcam,
            image_path=image_path,
            transform=transform,
            target_class=int(sample["y_pred"]),
            device=device,
        )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        axes[0, col].imshow(
            image,
            cmap="gray",
        )

        axes[0, col].set_title(
            f"{sample['Folder']}\n"
            f"P(PNEU): {sample['y_prob']:.3f}"
        )

        axes[0, col].axis("off")

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        axes[1, col].imshow(
            image,
            cmap="gray",
        )

        axes[1, col].imshow(
            heatmap,
            cmap="jet",
            alpha=0.4,
        )

        axes[1, col].set_title(
            "Grad-CAM"
        )

        axes[1, col].axis("off")

    axes[0, 0].set_ylabel(
        "Original",
        fontsize=12,
    )

    axes[1, 0].set_ylabel(
        "Grad-CAM",
        fontsize=12,
    )

    fig.suptitle(
        title_prefix,
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Cleanup
# ============================================================

def remove_hooks(gradcam):
    """
    Remove registered Grad-CAM hooks.
    """

    if gradcam is None:
        return

    if gradcam.forward_handle is not None:
        gradcam.forward_handle.remove()
        gradcam.forward_handle = None


def cleanup_gradcam(gradcam, model=None):
    """
    Remove hooks and release model resources.
    """

    remove_hooks(gradcam)

    if model is not None:
        del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()