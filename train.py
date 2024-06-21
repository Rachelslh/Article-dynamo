
from omegaconf import OmegaConf

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

model = TransformerDecoder(num_embeddings=train_dataset.vocab_size, embedding_dim=config['model']['embed_dim'])