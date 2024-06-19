from torch.utils.data import random_split

from dataset import TokenDataset


train_dataset = TokenDataset(train_path, **config)
val_dataset = TokenDataset(val_path, **config)

#n = int(split_perc * len(dataset))
#train_dataset, val_dataset = random_split(dataset, (n, len(dataset) - n))