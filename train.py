
import torch
from going_moduler import data_setup, model_builder, train_engine, utils
from torchvision import transforms
import os

train_dataloader, test_dataloader, valid_dataset, class_names= data_setup.create_dataloader(
    train_path, test_path, valid_path,data_transform, data_transform, os.cpu_count(),16)

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

model=model_builder.TinyVGG(3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)

adam=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_fn=torch.nn.CrossEntropyLoss()

results=train_engine.train(model=model,
                         train_dataloader=train_dataloader,
                         valid_dataloader=test_dataloader,
                         optimizer=adam,
                         loss_fn=loss_fn,
                         epochs=NUM_EPOCHS,
                         device=device)
