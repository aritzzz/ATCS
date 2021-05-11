import copy
import torch
import torch.nn as nn
from data_utils import *
from models import Classifier
from collections import defaultdict

class MetaTrainer(object):


    def __init__(self, model, train_loaders, val_loaders, num_episodes):
        self.outer_model = model 
        self.train_loaders = train_loaders #dataloaders for each of the meta-task
        #trainloader for a task contains n trainloaders where n is the number of classes
        self.val_loaders = val_loaders
        self.num_episodes = num_episodes
        self.n_tasks = len(self.train_loaders)
        self.loss_funcs = {
                    0: nn.CrossEntropyLoss(),
                    1: nn.CrossEntropyLoss(),
                    2: nn.CrossEntropyLoss()
                }
        self.task_classes = {
                    0: 3,
                    1: 2,
                    2: 2
                }
        self.outer_optimizer = torch.optim.AdamW(self.outer_model.encoder.parameters(),
                                                weight_decay=1e-4)

    
    def sample(self):
        """sample support set and query set for each episode
           returns:
           source_set: {task1: [class_1: K examples, class_2: K examples ...], task2: []}
           query_set: {task1: [class_1: K examples, class_2: K examples ...], task2: []}
        """
        ##HELP: what to do if any loader gets exhausted
        support_set = defaultdict(list)
        query_set = defaultdict(list)
        for task in range(self.n_tasks):
            support_task_batch = []
            query_task_batch = []
            for loader in self.train_loaders[task]:
                support_task_batch.append(next(loader))
            support_set[task] = support_task_batch

            for loader in self.val_loaders[task]:
                query_task_batch.append(next(loader))
            query_set[task] = query_task_batch
        
        return support_set, query_set

    
    def train(self):
        """Run episodes and perform outerloop updates """
        for epoch in range(3):
            print("---- starting epoch {} ----".format(epoch))
            self.outer_optimizer.zero_grad()
            for episode in range(self.num_episodes):
                source_sets, query_sets = self.sample()
                self.train_episode(source_sets[episode], query_sets[episode], episode)
            self.outer_optimizer.step()


    def forward(self, model, batch):
        input_ids, token_type_ids, attention_mask, _ = batch
        logits = model(input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)
        return logits

   
    def init_prototype_parameters(self, model, batch, n):
        input_ids, token_type_ids, attention_mask, _ = batch
        model.gamma = self.outer_model.encoder(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)["last_hidden_state"][:n,0,:]
    

    def inner_loop(self, model, source_set, loss_func):
        model.zero_grad()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-4)
        for batch in source_set:
            labels = batch[-1]
            optimizer.zero_grad()
            logits = self.forward(model, batch)
       
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()

    
    def calc_validation_grads(self, model, query_set, loss_func):
        predictions = []
        all_labels = []
        for batch in query_set:
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

    
    def train_episode(self, source_set, query_set, task):
        "train inner model for 1 step, returns gradients of encoder on support set."
        n_classes = self.task_classes[task]
        loss_func = self.loss_funcs[task]

        # Step 2: Duplicate model
        inner_model = copy.deepcopy(self.outer_model)

        # Step 3: Init prototype vectors (for now just take n embedding vectors).
        self.init_prototype_parameters(inner_model, source_set[0], n_classes)

        # Step 4: Init output parameters (phi).
        inner_model.init_phi(n_classes)

        # Step 5: perform k inner loop steps.
        self.inner_loop(inner_model, source_set, loss_func)
            
        # Step 6: Replace output parameters with trick.
        inner_model.replace_phi()

        # Step 7: Apply trained model on query set.
        self.calc_validation_grads(inner_model, query_set, loss_func)



if __name__ == "__main__":
    mnlitrain = MNLI.read(path='data/multinli_1.0/', split='train', slice_=1000)
    mnlimetadataset = MetaDataset.Initialize(mnlitrain)
    mnliloaders = MetaLoader(mnlimetadataset).get_data_loader(mnlimetadataset.dataloaders())

    mnlidev = MNLI.read(path='data/multinli_1.0/', split='dev_matched', slice_=1000)
    mnlimetadev = MetaDataset.Initialize(mnlidev, K=1)
    mnliloadersdev = MetaLoader(mnlimetadev).get_data_loader(mnlimetadev.dataloaders())


    SDtrain = StanceDataset.read(path='data/Stance/', split='train', slice_=1000)
    SDmetadataset = MetaDataset.Initialize(SDtrain)
    SDloaders = MetaLoader(SDmetadataset).get_data_loader(SDmetadataset.dataloaders())


    SDdev = StanceDataset.read(path='data/Stance', split='test', slice_=1000)
    SDmetadev = MetaDataset.Initialize(SDdev, K=1)
    SDloadersdev = MetaLoader(SDmetadev).get_data_loader(SDmetadev.dataloaders())

    
    #Paratrain = ParaphraseDataset.read(path='data/msrp/', split='train', slice_=1000)
    #ParaMetatrain = MetaDataset.Initialize(Paratrain, K=1)
    #Paraloaderstrain = MetaLoader(ParaMetatrain).get_data_loader(ParaMetatrain.dataloaders())

    #Paradev = ParaphraseDataset.read(path='data/msrp/', split='test', slice_=1000)
    #ParaMetadev = MetaDataset.Initialize(Paradev, K=1)
    #Paraloadersdev = MetaLoader(ParaMetadev).get_data_loader(ParaMetadev.dataloaders())


    model = Classifier({"freeze_bert": False})
    #print(model)

    meta_trainer = MetaTrainer(
                                model = model,
                                train_loaders = [mnliloaders, SDloaders],
                                val_loaders = [mnliloadersdev, SDloadersdev],
                                num_episodes = 2
                            )
    
    meta_trainer.train()
