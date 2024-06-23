
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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

model.training_step(*next(iter(train_dataloader)))