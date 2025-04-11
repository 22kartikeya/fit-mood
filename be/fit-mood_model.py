import os
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Custom Dataset class
class FashionDataset(Dataset):
    def __init__(self, dataframe, img_dir, label_encoder, transform=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform(self.data['category'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Paths
train_csv = './train.csv'
val_csv = './val.csv'
img_dir = 'processed_images'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load data
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

# Combine labels for fitting LabelEncoder
all_labels = pd.concat([train_df['category'], val_df['category']])
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Dataset and Dataloader
train_dataset = FashionDataset(train_df, img_dir, label_encoder, transform=transform)
val_dataset = FashionDataset(val_df, img_dir, label_encoder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Accuracy function
def calculate_accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    val_accuracy = calculate_accuracy(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss / len(train_loader):.4f} | "
          f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), "fashion_model.pth")
print("✅ Model trained and saved as fashion_model.pth")