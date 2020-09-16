import torch 
import torch.nn as nn

class CNN (nn.Module):
    def __init__(self, ):
        super(CNN, self).__init__()  

        self.cnn_layer = nn.Sequential(            
            nn.Conv2d(3,6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(6,12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(12,15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential( 
            nn.Linear(735, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

