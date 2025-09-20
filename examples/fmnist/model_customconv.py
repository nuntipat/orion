import orion.nn as on

class LargeConvNet(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(1, 32, kernel_size=3, padding=1)
        self.act1 = on.SiLU()
        self.bn1 = on.AvgPool2d(2)
        
        self.conv2 = on.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = on.SiLU()
        self.bn2 = on.AvgPool2d(2)
        
        self.flatten = on.Flatten()
        
        self.fc1 = on.Linear(64 * 7 * 7, 128)
        self.act3 = on.SiLU()

        self.fc2 = on.Linear(128, num_classes)

    def forward(self, x): 
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        return self.fc2(x)
 
class SmallConvNet(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(1, 5, kernel_size=2, padding=0, stride=2)
        self.bn1 = on.BatchNorm2d(5)
        self.act1 = on.Quad()
        
        self.fc1 = on.Linear(980, 100)
        self.bn2 = on.BatchNorm1d(100)
        self.act2 = on.Quad()
        
        self.flatten = on.Flatten()
        self.fc2 = on.Linear(100, num_classes)

    def forward(self, x): 
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.act2(self.bn2(self.fc1(x)))
        return self.fc2(x)