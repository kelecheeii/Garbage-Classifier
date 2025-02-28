import torch
from torchvision import transforms
from Final_Classifier import MultiModalClassifier  # Import model
from Final_Classifier import test_loader  # Import test_loader

# Define model & load best weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4  # Ensure this matches your training setup

# Load tokenizer (if needed for text processing)
model = MultiModalClassifier(num_classes=num_classes)  # Initialize model
model.load_state_dict(torch.load("/home/kelechi.mbibi/CLUSTER/Best_Classifier.pth", map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

# Define loss function for evaluation
criterion = torch.nn.CrossEntropyLoss()

# Function to evaluate the model on the entire test set
def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
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
    return total_loss / len(test_loader), accuracy

# Run evaluation on the test set
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

# Print test results
print(f"\ Test Loss: {test_loss:.4f}")
print(f" Test Accuracy: {test_acc:.2f}")
