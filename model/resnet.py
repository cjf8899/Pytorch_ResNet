import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    
    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
#         print(out.shape)
        return out
    
    
class ResNet(nn.Module):
    
    def __init__(self, Block, num_blocks, num_class=100):
        super(ResNet, self).__init__()

        self.Conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        print(num_blocks[0])
        self.layer1 = self.make_layer(Block, 64, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(Block, 64, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(Block, 128, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(Block, 256, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512, num_class)

    def make_layer(self, Block, in_planes, planes, num_blocks, stride):
        layers = []
        layers.append(Block(in_planes, planes, stride))
        i=0
        for i in range(int(num_blocks)-1):
            layers.append(Block(planes, planes, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.Conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    
    
def ResNet34():
    return ResNet(Block, [3,4,6,3])
