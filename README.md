# 🧠 LSB Steganalysis with CNN and Laplacian Filtering

This project presents a deep learning approach to detect LSB (Least Significant Bit) steganography in RGB PNG images. It leverages the ResNet18 architecture enhanced with Laplacian preprocessing and robustness-focused augmentations (noise, blur, etc.) to improve generalization under real-world distortions.

---

## 🛠️ Features

- ✅ **ResNet18 CNN model**
- ✅ **Laplacian filtering** to emphasize subtle pixel-level variations introduced by LSB embedding
- ✅ **Custom data pipeline** with Laplacian applied post-augmentation
- ✅ **Train-time augmentations**: Gaussian noise, salt & pepper, blur, flip
- ✅ **Evaluation tools**: Confusion matrix, per-image prediction
- ✅ **Manual testing support** for individual image prediction and visualization

---

## 🧪 Datasets

### 🗂️ stegoimagesdataset (used for training, validation and testing)
- **Name**: StegoImagesDataset by Marco Zuppelli  
- **Link**: [Kaggle - stegoimagesdataset](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset)  
- **Details**: Contains logo-type images in clean and LSB-stego versions.

**Data Breakdown:**

| Category              | Train | Test | Validation |
|-----------------------|-------|------|------------|
| Clean                 | 4000  | 2000 | 2000       |
| JavaScript            | 2363  | 1188 | 1214       |
| JavaScript in HTML    | 2284  | 1167 | 1162       |
| PowerShell            | 2468  | 1164 | 1213       |
| Ethereum Addresses    | 2473  | 1247 | 1193       |
| URL/IP addresses      | 2412  | 1234 | 1218       |
| **Total**             | 16000 | 8000 | 8000       |

---

### 🌄 otherdataset (used for cross-dataset generalization test)
- **Name**: Digital Steganography Dataset by Diego Zanchettin  
- **Link**: [Kaggle - digital-steganography](https://www.kaggle.com/datasets/diegozanchett/digital-steganography)  
- **Usage**: Only the `lsb` folder (excluding grayscale images) was used.  
- **Purpose**: Evaluate model performance on a different image domain (landscapes instead of logos).

---

## 🖼️ Data Format

- Format: .png
- Color mode : RGB
- Size: 256x256 pixels
- Dataset: 50% cover images, 50% stego images

---

## 🏋️ Training

Training Highlights:
- **GPU Utilization**: CUDA-enabled GPU for faster training
- **Epochs**: 100
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Monitors validation loss with a patience of 5 epochs
- **Adaptive Learning Rate**: Reduces learning rate on plateau with a factor of 0.5

---

## 📊 Evaluation

Evaluation tools include:
- **Confusion Matrix**: Visualizes the performance of the model on the test set
- **Per-Image Prediction**: Allows for manual testing of individual images


--- 
## 📚 References

- [ResNet Architecture](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [Laplacian Filtering](https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)