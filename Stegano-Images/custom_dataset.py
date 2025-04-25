import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from scipy.stats import kurtosis, skew
from torchvision import transforms
import warnings

class FusionFeatureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: dossier contenant deux sous-dossiers 'clean' et 'stego'
        transform: transformations à appliquer à l'image (ex: resize, normalize)
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label_name in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label_name)
            label = 0 if label_name.lower() == 'clean' else 1

            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def extract_stat_features(self, img):
      img_gray = img.convert('L')
      data = np.asarray(img_gray).astype(np.float32).flatten()

      data = np.clip(data, 1e-5, 255)
      eps = 1e-5

      try:
        std = np.std(data)
        range_val = np.max(data) - np.min(data)
        median = np.median(data)
        geo_median = np.exp(np.mean(np.log(data + eps)))  # safe log
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            skewness = skew(data)
            kurt = kurtosis(data)

        d1 = np.diff(data)
        d2 = np.diff(d1)

        var0 = np.var(data) + eps
        var1 = np.var(d1) + eps
        var2 = np.var(d2) + eps

        mobility = np.sqrt(var1 / var0)

        raw_complexity = (var2 / var1) - (var1 / var0)
        raw_complexity = np.maximum(raw_complexity, 0)
        complexity = np.sqrt(raw_complexity)

        features = np.array([std, range_val, median, geo_median, skewness, kurt, mobility, complexity], dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(features, dtype=torch.float32)
      
      except Exception as e:
        print("⚠️ Erreur lors de l'extraction des features statistiques :", e)
        return torch.zeros(8, dtype=torch.float32)
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        img = Image.open(img_path)

        stat_features = self.extract_stat_features(img)

        if self.transform:
            img = self.transform(img)

        return img, stat_features, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    # Exemple d'utilisation
    root_dir = 'stegoimagesdataset/train/train/'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = FusionFeatureDataset(root_dir=root_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for images, stat_features, labels in dataloader:
        print(images.shape, stat_features.shape, labels.shape)
        break  # Just to show the first batch