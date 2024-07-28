import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.


  """
  # put model on train
  model.train()

  # define loss and accuracy
  train_loss, train_acc=0.0, 0.0

  # iterate over dataloader

  for batch, (X, y) in enumerate(dataloader):
    # send data to device
    X, y = X.to(device), y.to(device)

    # forward pass

    y_pred=model(X)

    # loss
    loss=loss_fn(y_pred, y)
    train_loss+=loss.item()

    # zero_out grad
    optimizer.zero_grad()

    # backward
    loss.backward()

    # optimizer step
    optimizer.step()

    y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
    train_acc+= (y_pred_class==y).sum().item()/len(y_pred)

  train_loss/=len(dataloader)
  train_acc/=len(dataloader)
  return train_loss, train_acc



def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    test step for 1 epoch
    """

    model.eval()

    test_loss, test_acc=0.0, 0.0

    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      test_pred=model(X)

      loss=loss_fn(test_pred, y)
      test_loss+=loss.item()

      test_pred_class=torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
      test_acc+= (test_pred_class==y).sum().item()/len(test_pred)

    test_loss/=len(dataloader)
    test_acc/=len(dataloader)
    return test_loss, test_acc



def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """
  Train a PyTorch model

  """

  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc=train_step(model=model,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      device=device)
    test_loss, test_acc=test_step(model=model,
                                  dataloader=valid_dataloader,
                                  loss_fn=loss_fn,
                                  device=device)

    print(f"Epoch: {epoch+1} | train_loss: {train_loss} | train_acc: {train_acc} | test_loss: {test_loss} | test_acc: {test_acc}")
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
