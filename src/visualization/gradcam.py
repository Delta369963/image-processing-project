import cv2
import torch
import numpy as np
import uuid

from PIL import Image

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from src.configs.config import (
    CHECKPOINTS_DIR
)

from src.models.classification.resnet import (
    build_resnet18
)

from src.utils.transforms import (
    val_transforms
)

# =========================================================
# FORCE CPU FOR GRADCAM
# =========================================================

DEVICE = "cpu"

# =========================================================
# LOAD MODEL
# =========================================================

model = build_resnet18()

checkpoint_path = (
    CHECKPOINTS_DIR /
    "best_resnet18.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

# =========================================================
# TARGET LAYER
# =========================================================

target_layers = [
    model.layer4[-1]
]

# =========================================================
# GRADCAM OBJECT
# =========================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# =========================================================
# GENERATE HEATMAP
# =========================================================

def generate_gradcam(image_path):

    # =====================================================
    # UNIQUE OUTPUT NAME
    # =====================================================

    filename = (
        f"gradcam_{uuid.uuid4()}.jpg"
    )

    output_path = (
        f"outputs/heatmaps/{filename}"
    )

    # =====================================================
    # LOAD ORIGINAL IMAGE
    # =====================================================

    original_image = Image.open(
        image_path
    ).convert("RGB")

    resized_image = original_image.resize(
        (224, 224)
    )

    rgb_image = np.array(
        resized_image
    ).astype(np.float32) / 255.0

    # =====================================================
    # TRANSFORM
    # =====================================================

    transformed_image = val_transforms(
        original_image
    )

    input_tensor = (
        transformed_image
        .unsqueeze(0)
        .to(DEVICE)
    )

    # =====================================================
    # ENABLE GRADIENTS
    # =====================================================

    input_tensor.requires_grad = True

    # =====================================================
    # GENERATE CAM
    # =====================================================

    grayscale_cam = cam(
        input_tensor=input_tensor
    )

    grayscale_cam = grayscale_cam[0]

    # =====================================================
    # OVERLAY
    # =====================================================

    visualization = show_cam_on_image(

        rgb_image,

        grayscale_cam,

        use_rgb=True
    )

    # =====================================================
    # SAVE IMAGE
    # =====================================================

    visualization = cv2.cvtColor(

        visualization,

        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(

        output_path,

        visualization
    )

    # =====================================================
    # RETURN PATH
    # =====================================================

    return output_path