# data_load.py
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os


class Dataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.samples = []
        self.img_size = img_size
        
        if not os.path.exists(data_dir):
            raise RuntimeError(f"Directory not found: {data_dir}")
        
        try:
            valid_pairs = 0
            skipped_pairs = 0
            
            for slice_folder in sorted(os.listdir(data_dir)):
                if not slice_folder.startswith('Training'):
                    continue
                    
                slice_path = join(data_dir, slice_folder)
                
                if not os.path.isdir(slice_path):
                    continue
                    
                flair_img = None
                seg_img = None
                
                for file in sorted(os.listdir(slice_path)):
                    if 'image' in file:
                        flair_img = join(slice_path, file)
                    elif 'mask' in file:
                        seg_img = join(slice_path, file)

                if flair_img and seg_img:
                    seg_image = plt.imread(seg_img)
                    if np.sum(seg_image) > 0:
                        self.samples.append((flair_img, seg_img))
                        valid_pairs += 1
                    else:
                        skipped_pairs += 1
            
            print(f"Found {valid_pairs} valid image pairs")
            print(f"Skipped {skipped_pairs} pairs with empty masks")
            
            if len(self.samples) == 0:
                raise RuntimeError(f"No valid image pairs found in {data_dir}")
                
        except Exception as e:
            print(f"Error during dataset initialization: {str(e)}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            flair_path, seg_path = self.samples[idx]
            
            flair_image = Image.open(flair_path).convert('L')
            seg_image = Image.open(seg_path).convert('L')
            
            flair_image = np.array(flair_image)
            seg_image = np.array(seg_image)
            
            flair_tensor = torch.FloatTensor(flair_image)
            seg_tensor = torch.FloatTensor(seg_image)
            
            if flair_tensor.max() > 1.0:
                flair_tensor = flair_tensor / 255.0
            
            seg_tensor = (seg_tensor > 0).float()
            
            flair_tensor = flair_tensor.unsqueeze(0)
            seg_tensor = seg_tensor.unsqueeze(0)
            
            resize_transform = transforms.Resize(self.img_size)
            flair_tensor = resize_transform(flair_tensor)
            seg_tensor = resize_transform(seg_tensor)
            
            return flair_tensor, seg_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            print(f"FLAIR path: {flair_path}, Seg path: {seg_path}")
            raise