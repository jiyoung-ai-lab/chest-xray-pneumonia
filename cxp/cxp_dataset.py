
# # ============================================================
# # Chest X-Ray Pneumonia - Dataset
# # ============================================================

# import pandas as pd

# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset
# from PIL import Image

# from cxp.cxp_config import CLASS_TO_IDX


# # ============================================================
# # Dataset
# # ============================================================

# class ChestXrayDataset(Dataset):

#     def __init__(
#         self,
#         dataframe,
#         data_dir,
#         transform=None,
#     ):
#         self.dataframe = dataframe.reset_index(drop=True)
#         self.data_dir = data_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, index):

#         row = self.dataframe.iloc[index]

#         image_path = (
#             self.data_dir
#             / row["Folder"]
#             / row["File"]
#         )

#         # ----------------------------------------------------
#         # RGB input
#         # ResNet50 pretrained model의 원래 입력 형식에 맞춤
#         # ----------------------------------------------------

#         image = Image.open(image_path).convert("RGB")

#         label = CLASS_TO_IDX[row["Folder"]]

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, label


# # ============================================================
# # Train / Validation Split
# # ============================================================

# def split_train_validation(
#     dataframe,
#     val_size=0.2,
#     random_state=42,
# ):
#     """
#     Split training data into stratified
#     train / validation sets.
#     """

#     train_df, val_df = train_test_split(
#         dataframe,
#         test_size=val_size,
#         random_state=random_state,
#         stratify=dataframe["Folder"],
#     )

#     return (
#         train_df.reset_index(drop=True),
#         val_df.reset_index(drop=True),
#     )