#importing the required libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#loading the CIFAR10 dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
batch_size = 64

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_set = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_set = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#creating the VGG19 model

VGG_19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG19_net(nn.Module):
  def __init__(self, in_channels = 3, no_of_classes = 10):
    super(VGG19_net, self).__init__()
    self.in_channels = in_channels
    self.no_of_classes = no_of_classes
    self.conv_layers = self.create_conv_layers(VGG_19)
    self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fully_connected_layers = nn.Sequential(
                                                nn.Linear(in_features = 512 * 1 * 1, out_features = 4096, bias = True),
                                                nn.ReLU(inplace = True),
                                                nn.Dropout(p = 0.5),
                                                nn.Linear(in_features = 4096, out_features = 4096, bias = True),
                                                nn.ReLU(inplace = True),
                                                nn.Dropout(p = 0.5),
                                                nn.Linear(in_features = 4096, out_features = self.no_of_classes),
                                              )

  def create_conv_layers(self, arch):
   layers = []
   in_channels = self.in_channels
   for layer in arch:
     if(type(layer) == int):
       out_channels = layer

       layers += [nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1
                          ),
                  nn.BatchNorm2d(layer),
                  nn.ReLU(inplace = True)
                 ]
       in_channels = layer
     elif(layer == 'M'):
       layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]

   return nn.Sequential(*layers)    

  def forward(self, x):
    x = self.conv_layers(x)
    x = self.avg_pool(x)
    #max pooling works too
    x = torch.flatten(x, start_dim = 1, end_dim = -1)
    x = self.fully_connected_layers(x)
    return x

#passing data and training the model

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 3e-4
no_of_epochs = 10

vgg = VGG19_net().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg.parameters(), lr = learning_rate)

for epoch in range(no_of_epochs):
  for image, label in train_set:
    image = image.to(device)
    label = label.to(device)
    output = vgg(image)
    loss = loss_function(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print("Epoch", epoch + 1, "completed!")

print("Training of VGG-19 neural network successfully completed!")

#what does the output look like

with torch.no_grad():
  for image, label in test_set:
    image = image.to(device)
    label = label.to(device)
    outputs = vgg(image)
    temp, pred = torch.max(outputs, dim = 1)
    print(temp[0])
    print(pred[0])
    print(label[0])
    break

#testing the model

correct = 0
incorrect = 0

with torch.no_grad():
  for image, label in test_set:
    image = image.to(device)
    label = label.to(device)
    outputs = vgg(image)
    temp, pred = torch.max(outputs, 1)
    correct += (pred == label).sum().item()
    incorrect += (pred != label).sum().item()

model_accuracy = (correct/(correct + incorrect)) * 100
print("Accuracy of the VGG-19 model : ", model_accuracy, "%", sep = "")