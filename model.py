from torch import nn

class diabetesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=8, out_features=40)
        self.layer_2 = nn.Linear(in_features=40, out_features=40)
        self.layer_3 = nn.Linear(in_features=40, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))