import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())
# print(type(train_dataset))

BATCH_SIZE=32
EPOCHS=20
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FRE=100

#加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

class Lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')
        self.conv1=nn.Conv2d(1,10,5) # input:(1,28,28) output:(10,24,24)
        # torch.nn.MaxPool2d(kernel_size: Union[T, Tuple[T, ...]], stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0, dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
        self.pool1=nn.MaxPool2d(2,2)
        
        self.conv2=nn.Conv2d(10,20,3) # input:(10,12,12) output:(20,10,10)
        self.pool2=nn.MaxPool2d(2,2)

        self.flat=nn.Flatten()

        # torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
        self.fc1=nn.Linear(20*5*5,84)
        self.fc2=nn.Linear(84,10)

    def forward(self,x):
        out=self.pool1(F.relu(self.conv1(x)))
        out=self.pool2(F.relu(self.conv2(out)))
        out=self.flat(out)
        # out=out.view(out.size()[0], -1)
        out=self.fc1(out)
        out=self.fc2(out)
        return out
print("---net init---")
net=Lenet5()
net=net.to(DEVICE)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def test():
    total_correct = 0
    avg_loss = 0.0
    for i, data in enumerate(test_loader):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        output = net(inputs)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(test_dataset)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().item(), float(total_correct) / len(test_dataset)))
print("--- start  training ---")
for epoch in range(EPOCHS):
    running_loss = 0.0
    total_correct = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        # counting accuracy
        pred = outputs.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % PRINT_FRE == PRINT_FRE-1:
            print('train: [epoch: %d,    batch: %5d]     loss: %.3f    accuracy: %.3f' %
                  (epoch + 1, i + 1, running_loss / PRINT_FRE, float(total_correct)/ (PRINT_FRE * BATCH_SIZE)  ))
            running_loss = 0.0
            total_correct=0
            test()
