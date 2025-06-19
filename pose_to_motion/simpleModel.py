import torch.nn as nn

class SimpleMotionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMotionModel, self).__init__()
        
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.model(x)
