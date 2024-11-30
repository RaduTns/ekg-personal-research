import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models  # Assuming torchvision is installed
from pathlib import Path
import numpy as np
import cv2


class ECGDataset(Dataset):
  """
  A custom dataset class for loading preprocessed ECG images from a folder structure.
  """

  def __init__(self, data_dir):
    """
    Args:
        data_dir (str): Path to the directory containing the dataset folders.
    """
    self.data_dir = Path(data_dir)
    self.image_paths = []
    self.labels = []
    self.label_encoder = LabelEncoder()

    # Extract class labels from subfolder names
    self.labels = [folder.name for folder in self.data_dir.iterdir() if folder.is_dir()]
    self.label_encoder.fit(self.labels)

    # Loop through subdirectories (assuming class labels)
    for class_folder in self.data_dir.iterdir():
      if class_folder.is_dir():
        class_label = class_folder.name  # Extract class label from folder name
        for image_path in class_folder.iterdir():
          if image_path.is_file():
            self.image_paths.append(str(image_path))
            self.labels.append(class_label)

  def __len__(self):
    """
    Returns the number of images in the dataset.
    """
    return len(self.image_paths)

  def __getitem__(self, idx):
    """
    Retrieves an image and its corresponding label at a specific index.

    Args:
        idx (int): The index of the image to retrieve.

    Returns:
        tuple: A tuple containing the loaded image (tensor) and its label.
    """
    if idx >= len(self.image_paths):
      print(f"Index {idx} is out of range. Number of images: {len(self.image_paths)}")
      return None, None

    image_path = self.image_paths[idx]
    image = cv2.imread(image_path).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1))

    if idx >= len(self.labels):
      print(f"Index {idx} is out of range. Number of labels: {len(self.labels)}")
      return None, None

    label = self.labels[idx]
    label_encoded = self.label_encoder.transform([label])[0]
    label_tensor = torch.tensor(label_encoded, dtype=torch.long)

    return image_tensor, label_tensor
