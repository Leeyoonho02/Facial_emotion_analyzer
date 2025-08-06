import os
# import pandas as pd
from PIL import Image           # for image loading 
import torch
from torch.utils.data import Dataset, DataLoader    # for data loading
from torchvision import transforms           # for image transformation

'''
1. Dataset Class
    It demands to make 3 essential methods:
        1) init     : Rood dir path, phase, transform, dataset&labels
        2) len      : the number of the samples of this dataset
        3) getitem  : get the sample of certain index
'''
class FER2013Dataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
        }
        self.idx_to_emotion = {v: k for k, v in self.emotion_map.items()}
        
        # get the paths and labels of the images
        data_path = os.path.join(self.root_dir, self.phase)                 # ex)./fer2013/train
        for emotion_name in os.listdir(data_path):
            emotion_folder_path = os.path.join(data_path, emotion_name)     # ex)./fer2013/train/angry
            if os.path.isdir(emotion_folder_path):
                label = self.emotion_map[emotion_name]
                for img_name in os.listdir(emotion_folder_path):
                    img_path = os.path.join(emotion_folder_path, img_name)  # ex)./fer2013/train/angry/image1.jpg
                    self.image_paths.append(img_path)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.image_paths)} images for {self.phase} phase.")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('L')  
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
'''
2. Image transforming
    Transformation pipelines for train/test phases
'''
train_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])     # (0.0~1.0 -> -1.0~1.0), (x - mean) / std

    # Data augmentation
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),     # 이미지를 -10도에서 +10도 사이로 무작위 회전시킵니다.
    # transforms.ColorJitter(brightness=0.2, contrast=0.2), # 밝기, 대비 등을 무작위로 조절
])
test_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

'''
3. Make&check the instances of Dataset and DataLoader
'''
data_root = './fer2013/'

train_dataset = FER2013Dataset(root_dir=data_root, phase='train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = FER2013Dataset(root_dir=data_root, phase='test', transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print("\n--- Check the batch in DataLoader ---")
for images, labels in train_loader:
    print(f"Batch images shape: {images.shape}")        # ex.(batch size, channel number, height, width)
    print(f"Batch labels shape: {labels.shape}")        # ex.(64,)
    print(f"First 5 labels in batch: {labels[:5]}")
    break # check only the first batch.

print("\nData loading pipeline successfully built.")
