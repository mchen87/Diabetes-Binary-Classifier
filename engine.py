import torch
from utils import accuracy_fn


def train_step(model: torch.nn.Module, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               X, y
):

    # Put model in train mode
    model.train()
    
    # 1. Forward pass
    y_logits = model(X).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
     # 2. Calculate loss
    loss = loss_fn(y_logits, y)
    acc = accuracy_fn(y_true=y, 
                      y_pred=y_pred)
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # 4. Loss backward
    loss.backward()
    # 5. Optimizer step
    optimizer.step()
    return loss, acc

def test_step(model: torch.nn.Module, 
              loss_fn: torch.nn.Module,
              X, y
):
    # Put model in evaluation mode
    model.eval()
    with torch.inference_mode():
         # 1. Forward pass
        test_logits = model(X).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Calculate test loss
        test_loss = loss_fn(test_logits, y)
        test_acc = accuracy_fn(y_true=y, 
                      y_pred=test_pred)
        return test_loss, test_acc
        
def train(model: torch.nn.Module, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          X, y, X1, y1):
    #training loop
    for epoch in range(epochs):
        loss, acc = train_step(model = model, 
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   X = X, y = y)
        test_loss, test_acc = test_step(model = model, 
                   loss_fn=loss_fn,
                   X = X1, y = y1)
        
        # print out logistics every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
    
        