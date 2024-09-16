import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.MNIST(root='/Users/kareemaayman/task13.2/data', train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root='/Users/kareemaayman/task13.2/data', train=False, transform=test_transform, download=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=0.5)  # Dropout layer
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    best_val_acc = 0
    patience = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total

        # Evaluate on validation set
        val_acc = evaluate(model, test_loader)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
        if epoch - best_epoch >= patience:
            break

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Train the model
train(model, train_loader, test_loader, criterion, optimizer)

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('best_model.pth'))
test_acc = evaluate(model, test_loader)
print(f'Test Accuracy: {test_acc:.2f}%')