#import the required libraries

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

#custom Dataset class to load the data
class load_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.dataset[idx][1]
        image = torch.reshape(image, (-1, 1))
        return image, target

#model class

class MNIST_Classifier(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.fc0 = nn.Linear(in_features=input_size, out_features=30, bias=True)
        self.fc1 = nn.Linear(in_features=30, out_features=20, bias=True)
        self.fc2 = nn.Linear(in_features=20, out_features=num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image):
        output = self.fc0(image)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output

#trainer and testing functions

def train_model(model, criterion, optimizer, scheduler, device, num_epochs, train_data, test_data, input_size=784):
    for epoch in range(num_epochs):
        model = model.to(device)
        model.train()
        train_loss:float = 0.0
        train_accuracy:float = 0.0
        test_loss:float = 0.0
        test_accuracy:float = 0.0
        data_size:int = 0
        for images, labels in train_data:
            images = images.to(device)
            labels = labels.to(device)
            images = torch.reshape(images, (-1, input_size))
            output = model(images)
            loss = criterion(output, labels)
            predictions = torch.max(output, 1)[1]
            correct = (predictions==labels).sum().item()
            train_accuracy += correct
            train_loss += loss.item() * len(labels)
            data_size += len(labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss /= data_size
        train_accuracy /= data_size
        train_accuracy *= 100
        test_loss, test_accuracy = eval_model(model=model, criterion=criterion, device=device, test_data=test_data, input_size=input_size)

        print(f"Epoch: {epoch+1:02}/{num_epochs:02} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.3f} | Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}")


def eval_model(model, criterion, device, test_data, input_size=784):
    model.eval()
    test_loss:float = 0.0
    test_accuracy:float = 0.0
    data_size:int = 0

    for images, labels in test_data:
        images = images.to(device)
        labels = labels.to(device)
        images = torch.reshape(images, (-1, input_size))
        output = model(images)
        loss = criterion(output, labels)
        predictions = torch.max(output, 1)[1]
        correct = (predictions==labels).sum().item()
        test_loss += loss.item() * len(labels)
        test_accuracy += correct
        data_size += len(labels)
    
    test_loss /= data_size
    test_accuracy /= data_size
    test_accuracy *= 100

    return test_loss, test_accuracy

#downloading data

image_transforms = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ]
                   )

batch_size = 32

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=image_transforms)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=image_transforms)

#creating iterable datasets

train_data = load_dataset(train_set)
test_data = load_dataset(test_set)

train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#training the model

input_size = 784
num_classes = 10
model = MNIST_Classifier(input_size=input_size, num_classes=num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 20
learning_rate = 4e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, num_epochs=num_epochs, train_data=train_data, test_data=test_data, input_size=input_size)