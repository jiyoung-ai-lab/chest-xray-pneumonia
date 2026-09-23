# ============================================================
# Chest X-Ray Pneumonia - Project Configuration
# ============================================================

from pathlib import Path


# ============================================================
# Environment Configuration
# ============================================================

def configure_environment(env):
    """
    Configure project paths according to the execution environment.

    Parameters
    ----------
    env : str
        "pc", "remote", or "colab"

    Returns
    -------
    dict
        Project configuration
    """

    # ========================================================
    # Project Root
    # ========================================================

    if env == "pc":

        project_root = Path(
            r"D:\DEV\AI\projects\chest-xray-pneumonia"
        )

        ailib_root = Path(
            r"D:\DEV\AI\src"
        )

    elif env == "remote":

        project_root = Path(
            "/workspace/projects/chest-xray-pneumonia"
        )

        ailib_root = Path(
            "/workspace/src"
        )

    elif env == "colab":

        project_root = Path(
            "/content/drive/MyDrive/AI/projects/chest-xray-pneumonia"
        )

        ailib_root = Path(
            "/content/drive/MyDrive/AI/src"
        )

    else:

        raise ValueError(
            f"Unsupported environment: {env}"
        )


    # ========================================================
    # Data
    # ========================================================

    data_root = (
        project_root / "data"
    )

    raw_data_dir = (
        data_root / "chest_xray"
    )

    processed_data_dir = (
        data_root / "chest_xray_processed"
    )

    train_dir = (
        processed_data_dir / "train"
    )

    val_dir = (
        processed_data_dir / "val"
    )

    test_dir = (
        processed_data_dir / "test"
    )


    # ========================================================
    # Processed Image Information
    # ========================================================

    image_info_csv = (
        processed_data_dir
        / "processed_image_info.csv"
    )


    # ========================================================
    # Processed Data ZIP
    # ========================================================

    processed_data_zip = (
        data_root
        / "chest_xray_processed.zip"
    )


    # ========================================================
    # Colab Local Runtime Data
    # ========================================================

    if env == "colab":

        local_data_root = Path(
            "/content"
        )

        local_processed_data_dir = (
            local_data_root
            / "chest_xray_processed"
        )

        local_train_dir = (
            local_processed_data_dir
            / "train"
        )

        local_val_dir = (
            local_processed_data_dir
            / "val"
        )

        local_test_dir = (
            local_processed_data_dir
            / "test"
        )

        dataset_train_dir = local_train_dir
        dataset_val_dir = local_val_dir
        dataset_test_dir = local_test_dir

    else:

        local_data_root = None
        local_processed_data_dir = None
        local_train_dir = None
        local_val_dir = None
        local_test_dir = None

        dataset_train_dir = train_dir
        dataset_val_dir = val_dir
        dataset_test_dir = test_dir


    # ========================================================
    # Project Output
    # ========================================================

    experiments_dir = (
        project_root / "experiments"
    )

    models_dir = (
        project_root / "models"
    )

    reports_dir = (
        project_root / "reports"
    )


    # ========================================================
    # Classes
    # ========================================================

    class_to_idx = {
        "NORMAL": 0,
        "PNEUMONIA": 1,
    }

    idx_to_class = {
        0: "NORMAL",
        1: "PNEUMONIA",
    }


    # ========================================================
    # Configuration
    # ========================================================

    config = {
        "ENV": env,

        "PROJECT_ROOT": project_root,
        "AILIB_ROOT": ailib_root,

        "DATA_ROOT": data_root,
        "RAW_DATA_DIR": raw_data_dir,
        "PROCESSED_DATA_DIR": processed_data_dir,

        "TRAIN_DIR": train_dir,
        "VAL_DIR": val_dir,
        "TEST_DIR": test_dir,

        "IMAGE_INFO_CSV": image_info_csv,
        "PROCESSED_DATA_ZIP": processed_data_zip,

        "LOCAL_DATA_ROOT": local_data_root,
        "LOCAL_PROCESSED_DATA_DIR": local_processed_data_dir,
        "LOCAL_TRAIN_DIR": local_train_dir,
        "LOCAL_VAL_DIR": local_val_dir,
        "LOCAL_TEST_DIR": local_test_dir,

        "DATASET_TRAIN_DIR": dataset_train_dir,
        "DATASET_VAL_DIR": dataset_val_dir,
        "DATASET_TEST_DIR": dataset_test_dir,

        "EXPERIMENTS_DIR": experiments_dir,
        "MODELS_DIR": models_dir,
        "REPORTS_DIR": reports_dir,

        "CLASS_TO_IDX": class_to_idx,
        "IDX_TO_CLASS": idx_to_class,
    }

    return config

# ============================================================
# Backward Compatibility
# ============================================================

CLASS_TO_IDX = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}