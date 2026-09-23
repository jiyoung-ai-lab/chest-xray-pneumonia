# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:
    """
    Grad-CAM for CNN-based classification models.
    """

    def __init__(
        self,
        model,
        target_layer,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )

    def _save_activation(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

        # Activation tensor에 직접 gradient hook 등록
        output.register_hook(
            self._save_gradient
        )

    def _save_gradient(
        self,
        gradient,
    ):
        self.gradients = gradient

    def generate(
        self,
        image,
        target_class,
    ):
        """
        Generate Grad-CAM heatmap.
        """

        self.model.eval()

        self.activations = None
        self.gradients = None

        # Frozen model에서도 Grad-CAM을 위해
        # 입력에서부터 gradient graph를 생성
        image = image.requires_grad_(True)

        output = self.model(image)

        score = output[:, target_class].sum()

        self.model.zero_grad()

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

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activations
        ).sum(dim=1)

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


# ============================================================
# Cleanup
# ============================================================

def remove_hooks(
    gradcam,
):
    """
    Remove registered hooks.
    """

    gradcam.forward_handle.remove()