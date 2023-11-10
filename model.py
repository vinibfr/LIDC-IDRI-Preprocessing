import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Define the encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define the decoder (expansive path)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        # Decoder
        x2 = self.decoder(x1)
        return x2

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

transform = transforms.Compose([transforms.ToTensor()])

image_paths = ["path/to/image1.png", "path/to/image2.png", ...]
mask_paths = ["path/to/mask1.png", "path/to/mask2.png", ...]

dataset = CustomDataset(image_paths, mask_paths, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet(in_channels=1, out_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for inputs, masks in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}')

# Assuming `model` is the trained U-Net model
input_image = Image.open("path/to/input_image.png")
input_image = transform(input_image)
input_image = input_image.unsqueeze(0)  # Add batch dimension
output_mask = model(input_image)
