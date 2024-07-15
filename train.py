
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import lightning
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import TokenDataset
from model import TransformerDecoder


config = OmegaConf.load("config/config.yaml")

block_size = config['model']['block_size']
train_dataset = TokenDataset(**config['data']['val'], block_size=block_size)
val_dataset = TokenDataset(**config['data']['val'], block_size=block_size)

train_dataloader = DataLoader(train_dataset, **config['dataloader'])
val_dataloader = DataLoader(train_dataset, **config['dataloader'])

model = TransformerDecoder(num_tokens=train_dataset.vocab_size, **config['model'])

trainer = lightning.Trainer(**config['trainer'])
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
 
sequences = model.generate(torch.tensor(train_dataset.encoding.encode('Backpropagation is'), device=config.model.device).repeat(2, 1), 5, 4, device=config.model.device)
print(list(train_dataset.encoding.decode(l) for l in sequences.tolist()))

num_samples = config['data']['train']['max_samples']
batch_size = config['dataloader']['batch_size']
epochs_array = np.arange(1, config['trainer']['max_epochs'] + 1)

# Plot and label the training and validation loss values
plt.plot(epochs_array, model.loss['train'], label='Training Loss')
plt.plot(epochs_array, model.loss['val'][1:], label='Validation Loss') # Avoiding the sanity check val step here
 
plt.title('Training and Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
 
plt.legend(loc='best')

plt.savefig('assets/loss.jpg')

plt.show()