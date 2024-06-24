
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import lightning

from dataset import TokenDataset
from model import TransformerDecoder


config = OmegaConf.load("config.yaml")

block_size = config['model']['block_size']
train_dataset = TokenDataset(**config['data']['val'], block_size=block_size)
val_dataset = TokenDataset(**config['data']['val'], block_size=block_size)

'''
Keeping this in case i decide to use one data source (.txt) for both train/val splits

#n = int(split_perc * len(dataset))
#train_dataset, val_dataset = random_split(dataset, (n, len(dataset) - n))
'''

train_dataloader = DataLoader(train_dataset, **config['dataloader'])
val_dataloader = DataLoader(train_dataset, **config['dataloader'])

model = TransformerDecoder(num_tokens=train_dataset.vocab_size, **config['model'])

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = lightning.Trainer(**config['trainer'])
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)