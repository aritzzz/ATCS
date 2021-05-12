import torch
import random
from torch.utils.data import IterableDataset, DataLoader, Sampler
from datasets import MNLIDataset, StanceDataset, ParaphraseDataset

class EpisodeDataLoader(IterableDataset):

  def __init__(self, k=8, datasets=[], batch_size=32):
    """
    Params:
      k: number of samples in support set
    """
    super(EpisodeDataLoader).__init__()
    self.k = k
    self.datasets = datasets
    weights = {d : d.train_size for d in datasets}
    self.proportions = {d.name : w /sum(weights.values()) for d,w in weights.items()}
    print(self.proportions)


  def __iter__(self):
    while True:
      random.shuffle(self.datasets)
      d = self.datasets[0]
      support = self._get_iter(d)
      query = self._get_iter(d, support=False)
      yield (support, query, d.name)
    
  def _get_iter(self, dataset, support=True):
    split = 'train' if support else 'val'
    return dataset.get_iter(split)
  
  @classmethod
  def create_dataloader(cls, k, datasets, batch_size):
    data = cls(k, datasets)
    return DataLoader(data, batch_size, collate_fn = lambda x: x)

if __name__ == "__main__":
  dataloader = EpisodeDataLoader.create_dataloader(k=8, datasets=[
    MNLIDataset(batch_size=8),
    ParaphraseDataset(batch_size=8),
    StanceDataset(batch_size=8)
  ], batch_size=16)
  
  for batch in dataloader:
    # print(batch)
    print('-'*40)
    for x in batch[0]:
      print(list(x))
    break