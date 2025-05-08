import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import torchvision.transforms as T

class MRIDatasetParquet(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform if transform else T.Compose([
            T.Grayscale(num_output_channels=1),    # Se imagem for RGB
            T.Resize((224, 224)),                  # Compatível com ViT 2D
            T.ToTensor(),                          # [1, H, W]
            T.Lambda(lambda x: x.repeat(3, 1, 1)),  # [3, H, W] (finge RGB)
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        label = int(row['label'])
        img_bytes = row['image.bytes']
        image = Image.open(io.BytesIO(img_bytes))
        image = self.transform(image)
        return image, label

