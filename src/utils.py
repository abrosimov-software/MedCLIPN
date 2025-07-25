import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ROCOv2CaptionsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, tokenizer=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['ID'] + ".jpg")
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        caption = self.data.iloc[idx]['Caption']
        caption = self.tokenize(caption)
        return image, caption
