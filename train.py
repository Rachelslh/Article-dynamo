
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import lightning
import matplotlib.pyplot as plt
import numpy as np

from dataset import TokenDataset
from model import TransformerDecoder


config = OmegaConf.load("config.yaml")

block_size = config['model']['block_size']
train_dataset = TokenDataset(**config['data']['val'], block_size=block_size)
val_dataset = TokenDataset(**config['data']['val'], block_size=block_size)

train_dataloader = DataLoader(train_dataset, **config['dataloader'])
val_dataloader = DataLoader(train_dataset, **config['dataloader'])

model = TransformerDecoder(num_tokens=train_dataset.vocab_size, **config['model'])

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = lightning.Trainer(**config['trainer'])
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
 
num_samples = config['data']['train']['max_samples']
batch_size = config['dataloader']['batch_size']
epochs = np.arange(1, config['trainer']['max_epochs'] + 1)

# Plot and label the training and validation loss values
plt.plot(epochs, model.loss['train'], label='Training Loss')
plt.plot(epochs, model.loss['val'][1:], label='Validation Loss') # Avoiding the sanity check val step here
 
plt.title('Training and Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')

plt.xticks(np.arange(0, epochs, 2))
 
plt.legend(loc='best')
plt.show()

plt.savefig('loss.jpg')