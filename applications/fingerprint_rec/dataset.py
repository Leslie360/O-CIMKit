import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuration defaults
CROP_HEIGHT = 64
RESIZE_SIZE = (224, 224)
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]
BATCH_SIZE = 128

class CropBottom:
    """Crops the bottom N rows of the image."""
    def __init__(self, crop_height=CROP_HEIGHT):
        self.crop_height = crop_height

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, h - self.crop_height))

def get_dataloaders(root_dir=None, batch_size=BATCH_SIZE):
    if root_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root_dir = os.path.join(project_root, "data", "datasets", "fingerprint")
        
    train_transform = transforms.Compose([
        CropBottom(CROP_HEIGHT),
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_transform = transforms.Compose([
        CropBottom(CROP_HEIGHT),
        transforms.Resize(RESIZE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ Fingerprint dataset 'train' folder not found at: {train_dir}")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, test_loader
