from data_setup import split_data
from model import diabetesModel
from engine import train
import torch
from torch import nn

X_train, y_train, X_test, y_test = split_data()

model_0 = diabetesModel()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params= model_0.parameters(), lr = 0.1)

train(model = model_0, loss_fn=loss_fn, optimizer=optimizer, epochs= 1000, X= X_train, X1= X_test, y=y_train, y1=y_test)
y_pred = model_0(X_test)