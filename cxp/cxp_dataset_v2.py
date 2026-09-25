from ailib.dataset_v2 import ImageDataset


def split_train_validation(
    dataframe,
    val_size=0.2,
    random_state=42,
    stratify_column=None,
):
    """
    Split a dataframe into train and validation sets.
    """

    from sklearn.model_selection import train_test_split

    stratify = None

    if stratify_column is not None:
        stratify = dataframe[stratify_column]

    train_df, val_df = train_test_split(
        dataframe,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
    )


def create_transform(
    image_size,
    mean,
    std,
    augmentation=None,
):
    """
    Create image transform pipeline.
    """

    from torchvision import transforms

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    transform_list = [
        transforms.Resize(image_size),
    ]

    if augmentation is not None:
        transform_list.extend(augmentation)

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    return transforms.Compose(transform_list)


def create_datasets(
    train_dataframe,
    test_dataframe,
    train_dir,
    val_dir,
    test_dir,
    class_to_idx,
    image_size,
    normalization,
    split=True,
    val_size=0.2,
    random_state=42,
    stratify_column=None,
    train_augmentation=None,
):
    """
    Create train / validation / test datasets.

    Parameters
    ----------
    train_dataframe : pandas.DataFrame or dict
        If split=True:
            DataFrame containing the full training data.

        If split=False:
            Dictionary containing:
                {
                    "train": train_df,
                    "val": val_df,
                }

    test_dataframe : pandas.DataFrame
        Test metadata.

    train_dir : str or Path
        Training image root directory.

    val_dir : str or Path
        Validation image root directory.

    test_dir : str or Path
        Test image root directory.

    class_to_idx : dict
        Class-to-index mapping.

    image_size : int or tuple
        Target image size.

    normalization : dict
        {
            "mean": [...],
            "std": [...],
        }

    split : bool
        Whether to split train_dataframe into train / validation.

    val_size : float
        Validation ratio when split=True.

    random_state : int
        Random seed for splitting.

    stratify_column : str or None
        Column used for stratified splitting.

    train_augmentation : list or None
        Optional training augmentation transforms.

    Returns
    -------
    dict
        {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
    """

    # ----------------------------------------
    # 1. Normalization
    # ----------------------------------------

    mean = normalization["mean"]
    std = normalization["std"]

    # ----------------------------------------
    # 2. Transform
    # ----------------------------------------

    train_transform = create_transform(
        image_size=image_size,
        mean=mean,
        std=std,
        augmentation=train_augmentation,
    )

    eval_transform = create_transform(
        image_size=image_size,
        mean=mean,
        std=std,
        augmentation=None,
    )

    # ----------------------------------------
    # 3. Train / Validation
    # ----------------------------------------

    if split:

        train_df, val_df = split_train_validation(
            dataframe=train_dataframe,
            val_size=val_size,
            random_state=random_state,
            stratify_column=stratify_column,
        )

    else:

        train_df = train_dataframe["train"]
        val_df = train_dataframe["val"]

    # ----------------------------------------
    # 4. Dataset
    # ----------------------------------------

    train_dataset = ImageDataset(
        dataframe=train_df,
        data_dir=train_dir,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )

    val_dataset = ImageDataset(
        dataframe=val_df,
        data_dir=val_dir,
        class_to_idx=class_to_idx,
        transform=eval_transform,
    )

    test_dataset = ImageDataset(
        dataframe=test_dataframe,
        data_dir=test_dir,
        class_to_idx=class_to_idx,
        transform=eval_transform,
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }