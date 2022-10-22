#importing the required libraries

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#loading the CIFAR10 dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
batch_size = 32

train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)

test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

train_set = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

test_set = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

#creating the recurring blocks of the ResNet model

#Basic blocks for ResNets with 18/34 layers
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, intermediate_channels, stride = 1):
        super(BasicBlock, self).__init__()

        self.stride = stride

        self.conv_layer_1 = nn.Conv2d(in_channels, intermediate_channels * self.expansion, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.batch_norm_1 = nn.BatchNorm2d(intermediate_channels)

        self.conv_layer_2 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.batch_norm_2 = nn.BatchNorm2d(intermediate_channels)

        self.identity_change_dim = nn.Sequential()

        if stride != 1 or in_channels != intermediate_channels * self.expansion:
            self.identity_change_dim = nn.Sequential(
                                            nn.Conv2d(in_channels, intermediate_channels * self.expansion, kernel_size = 1, stride = stride, padding = 0, bias = False),
                                            nn.BatchNorm2d(intermediate_channels * self.expansion)
                                            )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity_x = x

        output = self.conv_layer_1(x)
        output = self.batch_norm_1(output)

        output = self.conv_layer_2(output)
        output = self.batch_norm_2(output)

        identity_x = self.identity_change_dim(identity_x)

        output += identity_x

        output = self.relu(output)

        return output

#Bottleneck blocks for ResNets with 50/101/152 layers
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, intermediate_channels, stride = 1):
        super(BottleneckBlock, self).__init__()

        self.stride = stride

        self.conv_layer_1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.batch_norm_1 = nn.BatchNorm2d(intermediate_channels)

        self.conv_layer_2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.batch_norm_2 = nn.BatchNorm2d(intermediate_channels)

        self.conv_layer_3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.batch_norm_3 = nn.BatchNorm2d(intermediate_channels * self.expansion)

        self.identity_change_dim = nn.Sequential()

        if stride != 1 or in_channels != intermediate_channels * self.expansion:
            self.identity_change_dim = nn.Sequential(
                                        nn.Conv2d(in_channels, intermediate_channels * self.expansion, kernel_size = 1, stride = stride, padding = 0, bias = False),
                                        nn.BatchNorm2d(intermediate_channels * self.expansion)
                                        )

        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity_x = x

        output = self.conv_layer_1(x)
        output = self.batch_norm_1(output)

        output = self.conv_layer_2(output)
        output = self.batch_norm_2(output)

        output = self.conv_layer_3(output)
        output = self.batch_norm_3(output)

        identity_x = self.identity_change_dim(identity_x)
        
        output += identity_x

        output = self.relu(output)

        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_of_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.expansion = 4

        self.conv_1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer_1 = self.make_layers(block, layers[0], 64, stride = 1)
        self.layer_2 = self.make_layers(block, layers[1], 128, stride = 2)
        self.layer_3 = self.make_layers(block, layers[2], 256, stride = 2)
        self.layer_4 = self.make_layers(block, layers[3], 512, stride = 2)

        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fully_connected_layer = nn.Linear(512 * block.expansion, num_of_classes)

        self.softmax = nn.Softmax(dim = 1)

    def make_layers(self, block, num_of_residual_blocks, intermediate_channels, stride):
        layers = []

        layers.append(block(self.in_channels, intermediate_channels, stride))

        self.in_channels = intermediate_channels * block.expansion

        for i in range(num_of_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_1(x)
        output = self.bn_1(output)
        output = self.relu_1(output)
        output = self.max_pool(output)

        output = self.layer_1(output)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.layer_4(output)

        output = self.average_pool(output)

        output = output.view(output.size(0), -1)

        output = self.fully_connected_layer(output)

        output = self.softmax(output)

        return output

def ResNet18(image_channels, num_of_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], image_channels, num_of_classes)

def ResNet101(image_channels, num_of_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], image_channels, num_of_classes)

def ResNet50(image_channels, num_of_classes):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], image_channels, num_of_classes)

def ResNet101(image_channels, num_of_classes):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], image_channels, num_of_classes)

def ResNet152(image_channels, num_of_classes):
    return ResNet(BottleneckBlock, [3, 8, 36, 3], image_channels, num_of_classes)

#passing data and training the model

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 3e-4
num_of_epochs = 20

net = ResNet152(image_channels = 3, num_of_classes = 10).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

for epoch in range(num_of_epochs):
    
    for image, label in train_set:
        image = image.to(device)
        label = label.to(device)

        output = net(image)

        loss = loss_function(output, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_of_epochs}: Training Loss: {loss.item()}')


print("Training of ResNet model successfully completed!")

#testing the model

correct = 0
incorrect = 0

with torch.no_grad():
  for image, label in test_set:
    image = image.to(device)
    label = label.to(device)
    outputs = net(image)
    temp, pred = torch.max(outputs, 1)
    correct += (pred == label).sum().item()
    incorrect += (pred != label).sum().item()

model_accuracy = (correct/(correct + incorrect)) * 100
print("Accuracy of the ResNet model : ", model_accuracy, "%", sep = "")