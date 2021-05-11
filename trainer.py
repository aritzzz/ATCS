from __future__ import division
from data_utils import *
from collections import defaultdict
from models import *
import torch.nn as nn





class MetaTrainer(object):
    def __init__(self, model, meta_loaders, train_loaders, val_loaders, num_episodes):
        self.base_model = model 
        self.train_loaders = train_loaders
        self.meta_loaders = meta_loaders
        self.val_loaders = val_loaders
        self.num_episodes = num_episodes
        self.n_tasks = len(self.train_loaders)
        self.inner_loop_updates = 1
        self.device = 'cpu'
        self.criterion = nn.CrossEntropyLoss() #It takes softmax internally

    
    def train(self):
         for episode in range(self.num_episodes):
            for task in range(self.n_tasks):
                self.inner_loop(task)
                break


    def inner_loop(self, task):
        for k in range(self.inner_loop_updates):
            prototypes = self._get_prototypes(task)
            for batch in self.train_loaders[task]:
                inp, labels = batch['input'], self._to_device(batch['label'])
                input_ids, token_type_ids, attention_mask = self._to_device(inp['input_ids']), self._to_device(inp['token_type_ids']), self._to_device(inp['attention_mask'])
                print(input_ids.shape, token_type_ids.shape, attention_mask.shape)
                encoding = self.base_model(input_ids, token_type_ids, attention_mask)
                pred = self._normalized_distance(encoding, prototypes)
                loss = self._get_loss(pred, labels)
                print(loss)
                break
        
             
    def _get_loss(self, predictions, true):
        return self.criterion(predictions, true)

    def train_batch(self, batch):
        pass

    def _get_prototypes(self, task):
        prototypes = defaultdict(lambda: self._to_device(torch.tensor([0.0]*768).unsqueeze(0)))
        for loader in self.meta_loaders[task]:
            count = 0
            for batch in loader:
                batch_ = batch['input']
                label = batch['label'][0]
                input_ids, token_type_ids, attention_mask = self._to_device(batch_['input_ids']), self._to_device(batch_['token_type_ids']), self._to_device(batch_['attention_mask'])
                encoding = self.base_model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask)
                prototypes[label] += torch.mean(encoding, dim=0, keepdim=True).detach()
                count+=1
            prototypes[label]/=count
        
        return {k:prototypes[k] for k in sorted(prototypes.keys())}
    
    def _to_device(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.tensor(inp)
        return inp.to(self.device)

    def _normalized_distance(self, repr_, prototypes):
        """
            repr_ : [bsz, emb_dim] shape tensor
            prototypes: {class_1: [1, emb_dim] tensor, class_2: [1, emb_dim] tensor, ... }
        """

        distance = torch.stack(
            tuple([torch.sum(torch.pow((repr_ - prototype), 2), dim=1) for _, prototype in prototypes.items()])
            )
        return -1*distance.T





if __name__ == "__main__":
    mnlitrain = MNLI.read(path='./multinli_1.0/', split='train', slice_=100)

    mnliload = DataLoader(mnlitrain, batch_size=32, collate_fn=collater)


    mnlimetadataset = MetaDataset.Initialize(mnlitrain)
    mnliloaders = MetaLoader(mnlimetadataset).get_data_loader(mnlimetadataset.dataloaders())

    # mnlidev = MNLI.read(path='./multinli_1.0/', split='dev_matched', slice_=1000)
    # mnlimetadev = MetaDataset.Initialize(mnlidev, K=1)
    # mnliloadersdev = mnlimetadev.dataloaders()


    SDtrain = StanceDataset.read(split='train', slice_=100)

    SDload = DataLoader(SDtrain, batch_size=32, collate_fn=collater)

    SDmetadataset = MetaDataset.Initialize(SDtrain)
    SDloaders = MetaLoader(SDmetadataset).get_data_loader(SDmetadataset.dataloaders())


    # SDdev = StanceDataset.read(split='test', slice_=1000)
    # SDmetadev = MetaDataset.Initialize(SDdev, K=1)
    # SDloadersdev = MetaLoader().get_data_loaderSDmetadev.dataloaders()


    config = {"num_classes": 3, 'freeze_bert': True}
    model = Classifier(config)
    print(model)

    meta_trainer = MetaTrainer(
                                model = model,
                                train_loaders = [mnliload, SDload],
                                meta_loaders = [mnliloaders, SDloaders],
                                val_loaders = None, #[mnliloadersdev, SDloadersdev],
                                num_episodes = 1
                            )
    
    meta_trainer.train()