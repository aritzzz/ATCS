import copy
import torch
import torch.nn as nn
from data_utils import *
from models import Classifier
from collections import defaultdict

class MetaTrainer(object):


    def __init__(self, model, train_loaders, meta_loaders, val_loaders, num_episodes):
        self.outer_model = model 
        self.train_loaders = train_loaders #dataloaders for each of the meta-task
        #trainloader for a task contains n trainloaders where n is the number of classes
        self.val_loaders = val_loaders
        self.meta_loaders = meta_loaders
        self.num_episodes = num_episodes
        self.n_tasks = len(self.train_loaders)
        self.device = torch.device("cpu")
        self.loss_funcs = {
                    0: nn.CrossEntropyLoss(),
                    1: nn.CrossEntropyLoss(),
                    2: nn.CrossEntropyLoss()
                }
        self.task_classes = {
                    0: 2,
                    1: 2,
                    2: 2
                }
        self.outer_optimizer = torch.optim.AdamW(self.outer_model.encoder.parameters(),
                                                weight_decay=1e-4)
    
    def train(self):
        """Run episodes and perform outerloop updates """
        for epoch in range(3):
            self.outer_optimizer.zero_grad()
            for episode in range(self.num_episodes):
                print("---- Starting episode {} of epoch {} ----".format(episode, epoch))
                self.train_episode(episode)
            self.outer_optimizer.step()


    def forward(self, model, batch):
        batch_ = batch['input']
        input_ids = self._to_device(batch_['input_ids'])
        token_type_ids = self._to_device(batch_['token_type_ids'])
        attention_mask = self._to_device(batch_['attention_mask'])
        logits = model(input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)
        return logits

    def _to_device(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.tensor(inp)
        return inp.to(self.device)

   
    def init_prototype_parameters(self, model, task):
        n_classes = self.task_classes[task]
        prototypes = {i: [] for i in range(n_classes)}
        for loader in self.meta_loaders[task]:
            for i, batch in enumerate(loader):
                batch_ = batch['input']
                input_ids = self._to_device(batch_['input_ids'])
                token_type_ids = self._to_device(batch_['token_type_ids'])
                attention_mask = self._to_device(batch_['attention_mask'])
        
                encoding = self.outer_model.encoder(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)["last_hidden_state"][:,0,:]
                label = batch['label'][0]
                prototypes[label].append(encoding)
        
        tmp = torch.zeros((n_classes, 768))
        for label, proto in prototypes.items():
            tmp[label,:] = torch.mean(torch.cat(proto, dim=0), dim=0)
        
        model.gamma = tmp
        

    def inner_loop(self, model, task):
        loss_func = self.loss_funcs[task]
        model.zero_grad()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-4)
        
        for batch in self.train_loaders[task]:
            labels = self._to_device(batch['label'])
            optimizer.zero_grad()
            logits = self.forward(model, batch)
            
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()

    
    def calc_validation_grads(self, model, task):
        loss_func = self.loss_funcs[task]
        predictions = []
        all_labels = []
        for i, batch in enumerate(self.val_loaders[task]):
            labels = batch[-1]
            logits = self.forward(model, batch)

            predictions.append(logits)
            all_labels.append(torch.unsqueeze(labels, dim=0))
        
        predictions = torch.cat(predictions, dim=0)
        all_labels = torch.cat(all_labels).squeeze()

        loss = loss_func(predictions, all_labels)

        grads_inner_model = torch.autograd.grad(outputs=loss,
                                            inputs=model.encoder.parameters(),
                                            retain_graph=True,
                                            create_graph=True,
                                            allow_unused=True)

        grads_outer_model = torch.autograd.grad(outputs=loss,
                                            inputs=self.outer_model.encoder.parameters(),
                                            allow_unused=True)

        for i, (name, param) in enumerate(self.outer_model.named_parameters()):
            if 'pooler' in name:
                continue
            elif param.grad is None:
                param.grad = grads_inner_model[i] + grads_outer_model[i]
            else:
                param.grad += grads_inner_model[i] + grads_outer_model[i]

    
    def train_episode(self, task):
        "train inner model for 1 step, returns gradients of encoder on support set."
        n_classes = self.task_classes[task]
        loss_func = self.loss_funcs[task]

        # Step 2: Duplicate model
        inner_model = copy.deepcopy(self.outer_model)

        # Step 3: Init prototype vectors (for now just take n embedding vectors).
        print("---- Initializing prototype parameters ----")
        self.init_prototype_parameters(inner_model, task)

        # Step 4: Init output parameters (phi).
        inner_model.init_phi(n_classes)

        # Step 5: perform k inner loop steps.
        print("---- Performing inner loop updates ----")
        self.inner_loop(inner_model, task)
            
        # Step 6: Replace output parameters with trick.
        inner_model.replace_phi()

        # Step 7: Apply trained model on query set.
        print("---- Calculating gradients on query set ----")
        self.calc_validation_grads(inner_model, task)



if __name__ == "__main__":
    
    config = {'freeze_bert': False}
    model = Classifier(config)

    SDtrain = StanceDataset.read(path='data/Stance/', split='train', slice_=100)
    SDload = DataLoader(SDtrain, batch_size=32, collate_fn=collater)
    SDmetadataset = MetaDataset.Initialize(SDtrain)
    SDloaders = MetaLoader(SDmetadataset).get_data_loader(SDmetadataset.dataloaders())

    SDdev = StanceDataset.read(path='data/Stance/', split='test', slice_=1000)
    SDmetadev = MetaDataset.Initialize(SDdev, K=1)
    SDloadersdev = MetaLoader(SDmetadev).get_data_loader(SDmetadev.dataloaders())
    
     
    meta_trainer = MetaTrainer(
                            model = model,
                            train_loaders = [SDload],
                            meta_loaders = [SDloaders],
                            val_loaders = [SDloadersdev],
                            num_episodes = 1
                            )
    
    meta_trainer.train()
    
    
    
    
    
    
    #datasetloader = EpisodeDataLoader.create_dataloader(k=8, datasets=[
    #        ParaphraseDataset(batch_size=8),
    #        StanceDataset(batch_size=8)
    #    ], batch_size=16)
    
   # datasetloader = EpisodeDataLoader(k=8, datasets=[
   #         StanceDataset(batch_size=8)
   #     ], batch_size=16)

    #para_train = datasetloader._get_iter(datasetloader.datasets[0], support=True)
    #para_dev = datasetloader._get_iter(datasetloader.datasets[0], support=False)

   # stance_train = datasetloader._get_iter(datasetloader.datasets[1], support=True)
   # stance_dev = datasetloader._get_iter(datasetloader.datasets[1], support=False)
   
   # SDmetadataset = MetaDataset.Initialize(stance_train)
   # SDloaders = MetaLoader(SDmetadataset).get_data_loader(SDmetadataset.dataloaders())

   # config = {'freeze_bert': False}
   # model = Classifier(config)

   # meta_trainer = MetaTrainer(
   #                             model = model,
   #                             train_loaders = [stance_train],
   #                             val_loaders = [stance_train],
   #                             num_episodes = 1
   #                         )
    
    #meta_trainer.train()
