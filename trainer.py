from data_utils import *
from collections import defaultdict

class MetaTrainer(object):
    def __init__(self, model, train_loaders, val_loaders, num_episodes):
        self.base_model = model 
        self.train_loaders = train_loaders #dataloaders for each of the meta-task
        #trainloader for a task contains n trainloaders where n is the number of classes
        self.val_loaders = val_loaders
        self.num_episodes = num_episodes
        self.n_tasks = len(self.train_loaders)

    
    def sample(self):
        """sample support set and query set for each episode
           returns:
           source_set: {task1: [class_1: K examples, class_2: K examples ...], task2: []}
           query_set: {task1: [class_1: K examples, class_2: K examples ...], task2: []}
        """
        ##HELP: what to do if any loader gets exhausted
        source_set = defaultdict(lambda: [])
        query_set = defaultdict(lambda: [])
        for task in range(self.n_tasks):
            source_task_batch = []
            query_task_batch = []
            for loader in self.train_loaders[task]:
                source_task_batch.append(next(loader))
            source_set[task] = source_task_batch

            for loader in self.val_loaders[task]:
                query_task_batch.append(next(loader))
            query_set[task] = query_task_batch
        
        return source_set, query_set


    
    def train(self):
         for episode in range(self.num_episodes):
             source_sets, query_sets = self.sample()
             print(query_sets)
             break



    

    def train_episode_step(self):
        pass


if __name__ == "__main__":
    mnlitrain = MNLI.read(path='./multinli_1.0/', split='train', slice_=1000)
    mnlimetadataset = MetaDataset.Initialize(mnlitrain)
    mnliloaders = MetaLoader(mnlimetadataset).get_data_loader(mnlimetadataset.dataloaders())

    mnlidev = MNLI.read(path='./multinli_1.0/', split='dev_matched', slice_=1000)
    mnlimetadev = MetaDataset.Initialize(mnlidev, K=1)
    mnliloadersdev = MetaLoader(mnlimetadev).get_data_loader(mnlimetadev.dataloaders())


    SDtrain = StanceDataset.read(split='train', slice_=1000)
    SDmetadataset = MetaDataset.Initialize(SDtrain)
    SDloaders = MetaLoader(SDmetadataset).get_data_loader(SDmetadataset.dataloaders())


    SDdev = StanceDataset.read(split='test', slice_=1000)
    SDmetadev = MetaDataset.Initialize(SDdev, K=1)
    SDloadersdev = MetaLoader(SDmetadev).get_data_loader(SDmetadev.dataloaders())


    meta_trainer = MetaTrainer(
                                model = None,
                                train_loaders = [mnliloaders, SDloaders],
                                val_loaders = [mnliloadersdev, SDloadersdev],
                                num_episodes = 2
                            )
    
    meta_trainer.train()