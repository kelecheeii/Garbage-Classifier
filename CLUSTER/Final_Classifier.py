# importing various pytorch libraries necessary for deep learning and processing our images
import torch
import torch.nn as nn
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import ResNet18_Weights
from torchvision import transforms, models
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import os
import re
import torch.optim as optim

# Checking for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define dataset paths
train_path = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
val_path = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"
test_path = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"

# Data augmentation: Adding extra transformations to reduce overfitting
torchvision_transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip(p=0.2),  
    transforms.RandomRotation(20),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

torchvision_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to extract text from filenames
def read_files_with_text_labels(path):
    texts, labels = [], []
    class_folders = sorted(os.listdir(path))
    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    processed_file_name, _ = os.path.splitext(file_name)
                    text = processed_file_name.replace('_', ' ')  
                    text_processed = re.sub(r'\d+', '', text)
                    texts.append(text_processed)
                    labels.append(label_map.get(class_name, -1))

    return np.array(texts), np.array(labels)

# Define dataset class
class MultiModalDataset(Dataset):
    def __init__(self, image_path, texts, labels, tokenizer, max_len, image_transform):
        self.image_path = image_path
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_transform = image_transform
        self.class_folders = sorted(os.listdir(image_path))  
        self.label_map = {class_name: idx for idx, class_name in enumerate(self.class_folders)}

        self.image_files = []
        for class_name in self.class_folders:
            class_dir = os.path.join(image_path, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    if os.path.isfile(file_path):
                        self.image_files.append((file_path, self.label_map[class_name]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        image_path, _ = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        return {
            'image': image,
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load datasets
train_texts, train_labels = read_files_with_text_labels(train_path)
val_texts, val_labels = read_files_with_text_labels(val_path)
test_texts, test_labels = read_files_with_text_labels(test_path)

train_dataset = MultiModalDataset(train_path, train_texts, train_labels, tokenizer, 24, torchvision_transform)
val_dataset = MultiModalDataset(val_path, val_texts, val_labels, tokenizer, 24, torchvision_transform)
test_dataset = MultiModalDataset(test_path, test_texts, test_labels, tokenizer, 24, torchvision_transform_test)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define Multi-Modal Classifier
class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(MultiModalClassifier, self).__init__()

        # Text Model (DistilBERT)
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 256)
        self.text_activation = nn.ReLU()
        self.text_norm = nn.LayerNorm(256)  # Implement Normalizing text features discussed in class
        self.text_dropout = nn.Dropout(dropout_rate)

        # Image Model (ResNet)
        self.image_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Unfreeze last two layers of ResNet for fine-tuning
        for param in self.image_model.layer3.parameters():
            param.requires_grad = True
        for param in self.image_model.layer4.parameters():
            param.requires_grad = True

        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 256)
        self.image_activation = nn.ReLU()
        self.image_norm = nn.BatchNorm1d(256)  # Implement Normalizing image features discussed in class
        self.image_dropout = nn.Dropout(dropout_rate)

        # Fusion layer
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask, image):
        # Process text features
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        text_features = self.text_fc(text_features)
        text_features = self.text_activation(text_features)
        text_features = self.text_norm(text_features)  # Apply LayerNorm here
        text_features = self.text_dropout(text_features)

        # Process image features
        image_features = self.image_model(image)
        image_features = self.image_activation(image_features)
        image_features = self.image_norm(image_features)  # Apply BatchNorm here
        image_features = self.image_dropout(image_features)

        # Merge text & image features
        combined_features = torch.cat((text_features, image_features), dim=1)

        return self.classifier(combined_features)

# Training Setup
num_classes = 4
model = MultiModalClassifier(num_classes=num_classes, dropout_rate=0.3).to(device)

# Optimizer & Learning Rate Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.00002)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

# defining our training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0  # we avoid division by zero errors
    return total_loss / len(train_loader), accuracy  # Return average loss & accuracy


# defining our evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation for validation
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(val_loader), accuracy

# Early Stopping Variables
best_loss = float("inf")
patience = 3
counter = 0

# Training loop (kept in **your structure**)
if __name__ == "__main__":
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}/20")

        # Train for one epoch
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)

        # Validate the model
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save model if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "Best_Classifier.pth")
            print("Model saved!")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
        
        scheduler.step()
