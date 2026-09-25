# ============================================================
# Chest X-Ray Pneumonia - Data
# ============================================================

from pathlib import Path

import pandas as pd

from ailib.image import ImageInspector, ImageAnalyzer


# ============================================================
# Image Inspection
# ============================================================

def inspect_dataset(data_dir):
    """
    Inspect the Chest X-Ray image dataset.
    """

    inspector = ImageInspector()

    return inspector.inspect_dataset(
        data_dir
    )


# ============================================================
# Image Analysis
# ============================================================

def analyze_dataset(dataframe):
    """
    Analyze image metadata and derived features.
    """

    analyzer = ImageAnalyzer()

    dataframe = analyzer.add_features(
        dataframe
    )

    return dataframe


# ============================================================
# Load Processed Image Information
# ============================================================

def load_image_info(csv_path):
    """
    Load processed image information.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"File not found: {csv_path}"
        )

    return pd.read_csv(csv_path)