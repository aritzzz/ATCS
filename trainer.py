from __future__ import division
from data_utils import *
from collections import defaultdict
from models import *
import torch.nn as nn





class MetaTrainer(object):
    def __init__(self, model, train_datasets, valid_datasets, num_episodes):
        self.base_model = model
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets 
        self.num_episodes = num_episodes
        self.n_tasks = len(self.train_datasets)
        # self.meta_loaders = meta_loaders

        self.train_loaders = defaultdict(lambda: [])
        self.valid_loaders = defaultdict(lambda: [])
        for task in range(self.n_tasks):
            self.train_loaders[task] = self._initialize_loaders(task)
        
        for task in range(self.n_tasks):
            self.valid_loaders[task] = self._initialize_loaders(task, valid=True)

        self.inner_loop_updates = 1
        self.device = 'cpu'
        self.criterion = nn.CrossEntropyLoss() #It takes softmax internally


    def _initialize_loaders(self, task, valid=False):
        if not valid:
            return self.train_datasets[task].get_dataloaders()
        else:
            return self.valid_datasets[task].get_dataloaders()
    

    def sample(self):
        """sample support set and query set for each episode
            returns:
            source_set: {task1: {class_1: {}, class_2: {}...}, task2: ...}
            query_set: {task1: {class_1: {}, class_2: {}...}, task2: ...}
    """
        source_set = defaultdict(lambda: {})
        query_set = defaultdict(lambda: {})
        for task in range(self.n_tasks):
            dl = self.train_loaders[task]
            source_task_batch = {}
            for i, loader in enumerate(dl):
                b = next(loader, -1)
                if b == -1:
                    self.train_loaders[task] = self._initialize_loaders(task)
                    b = next(self.train_loaders[task][i], -1)
                label = b["labels"][0].item()
                source_task_batch[label] = b
            source_set[task] = source_task_batch

        for task in range(self.n_tasks):
            dl = self.valid_loaders[task]
            query_task_batch = {}
            for i, loader in enumerate(dl):
                b = next(loader, -1)
                if b == -1:
                    self.valid_loaders[task] = self._initialize_loaders(task, valid=True)
                    b = next(self.valid_loaders[task][i], -1)
                label = b["labels"][0].item()
                query_task_batch[label] = b
            query_set[task] = query_task_batch
        return source_set, query_set


    def train(self):
        for episode in range(100):
            source, query = self.sample()
            print("-----episode----- {}".format(episode))
            print(source, query)
        #  for episode in range(self.num_episodes):
        #     for task in range(self.n_tasks):
        #         self.inner_loop(task)
        #         break


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
        
    def _init_prototype_parameters(self, prototypes):
        input_ids, token_type_ids, attention_mask, _ = batch
        model.gamma = self.outer_model.encoder(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)["last_hidden_state"][:n,0,:]

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

    # mnliload = DataLoader(mnlitrain, batch_size=32, collate_fn=collater)


    mnlimetadataset = MetaDataset.Initialize(mnlitrain)

    mnlidev = MNLI.read(path='./multinli_1.0/', split='dev_matched', slice_=1000)
    mnlimetadev = MetaDataset.Initialize(mnlidev, K=1)


    SDtrain = StanceDataset.read(split='train', slice_=100)

    # SDload = DataLoader(SDtrain, batch_size=32, collate_fn=collater)

    SDmetadataset = MetaDataset.Initialize(SDtrain)
    


    SDdev = StanceDataset.read(split='test', slice_=1000)
    SDmetadev = MetaDataset.Initialize(SDdev, K=1)
    # SDloadersdev = MetaLoader().get_data_loaderSDmetadev.dataloaders()


    config = {"num_classes": 3, 'freeze_bert': True}
    model = Classifier(config)
    print(model)

    meta_trainer = MetaTrainer(
                                model = model,
                                train_datasets = [mnlimetadataset, SDmetadataset],
                                valid_datasets = [mnlimetadev, SDdev],
                                num_episodes = 1
                            )
    
    meta_trainer.train()