from ailib.dataloader_v2 import create_dataloader


def create_dataloaders(
    datasets,
    batch_size,
    num_workers=0,
    pin_memory=False,
):
    """
    Create train / validation / test DataLoaders.

    Parameters
    ----------
    datasets : dict
        {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

    batch_size : int
        Batch size used for all DataLoaders.

    num_workers : int, default=0
        Number of DataLoader workers.

    pin_memory : bool, default=False
        Whether to use pinned memory.

    Returns
    -------
    dict
        {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
    """

    train_loader = create_dataloader(
        dataset=datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = create_dataloader(
        dataset=datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = create_dataloader(
        dataset=datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }