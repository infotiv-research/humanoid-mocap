import torch.nn as nn
import torch

class ImprovedMotionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[124, 64, 32, 16], dropout=0.3):
    # def __init__(self, input_size, output_size, hidden_sizes=[1024, 704, 960, 576], dropout=0.3940038235074028):
        super(ImprovedMotionModel, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        
        self.hidden_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()  
        
        for i in range(len(hidden_sizes) - 1):
        
            layer = nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(layer)           
 
            if hidden_sizes[i] != hidden_sizes[i + 1]:
                projection = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            else:
                projection = nn.Identity()
            self.residual_projections.append(projection)
        
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.apply(self._init_weights)
    # https://www.geeksforgeeks.org/xavier-initialization/
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = torch.nn.functional.gelu(x)

        for i, layer in enumerate(self.hidden_layers):
            residual = x
            x = layer(x)
            
          
            projected_residual = self.residual_projections[i](residual)
            x = x + projected_residual  

        return self.output_layer(x)

