import copy
import torch.nn as nn
import json
import argparse
from tqdm import tqdm
from data_utils import *
from models import Classifier
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

class Plotter(object):

  def __init__(self, name):
    self.name = name
    self.logger = defaultdict(lambda: []) #{'train': [], 'valid': []}

  def update(self, dict_):
    for k, v in dict_.items():
        self.logger[k].append(v)

  def plot(self):

    for k, v in self.logger.items():
        iters = range(len(self.logger[k]))
        plt.plot(iters, self.logger[k], c='dodgerblue', label="k")
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel(k, fontsize=12)
        # plt.title(self.name, fontsize=10)
        # plt.legend(loc="best", fontsize=12, frameon=False)
        plt.tight_layout()
        os.makedirs('./figs', exist_ok=True)
        plt.savefig('./figs/' + self.name + '_ ' + k + '.png')
        plt.show()



from transformers import AdamW, get_cosine_schedule_with_warmup

class MetaTrainer(object):


    def __init__(self, model, train_datasets,
                val_datasets, test_datasets,
                task_classes, epochs,
                inner_lr, outer_lr,
                inner_batch_size, num_episodes,
                model_save_path, results_save_path,
                clip_value, exp_name,
                seed=42, device=torch.device("cpu")):
        self.set_seed(seed)
        self.outer_model = model.to(device)
        self.n_tasks = num_episodes
        self.n_epochs = epochs
        self.model_save_path = model_save_path
        self.results_save_path = results_save_path
        self.train_datasets = train_datasets
        self.valid_datasets = val_datasets
        self.test_datasets = test_datasets
        self.exp_name = exp_name
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_batch_size = inner_batch_size

        os.makedirs(self.model_save_path, exist_ok = True)
        os.makedirs(self.results_save_path, exist_ok=True)

        self.train_loaders = defaultdict(lambda: [])
        self.valid_loaders = defaultdict(lambda: [])
        self.test_loaders = defaultdict(lambda: [])
        for task in range(self.n_tasks):
            self.train_loaders[task] = self._initialize_loaders(task)
            self.valid_loaders[task] = self._initialize_loaders(task, type_="valid")

        self.num_episodes = num_episodes
        self.device = device
        self.loss_funcs = {
                    0: nn.CrossEntropyLoss(),
                    1: nn.CrossEntropyLoss(),
                    2: nn.CrossEntropyLoss()
                }
        self.task_classes = task_classes
        self.clip_value = clip_value
        # self.outer_optimizer = torch.optim.AdamW(self.outer_model.encoder.parameters(),
        self.outer_optimizer = AdamW(self.outer_model.encoder.parameters(),
                                    self.outer_lr,  weight_decay=1e-4)
        self.outer_lr_scheduler = get_cosine_schedule_with_warmup(self.outer_optimizer, num_training_steps=self.n_epochs,  num_warmup_steps=int(0.10*self.n_epochs))

        self.inner_results = {"losses":defaultdict(list),
                    "accuracy":defaultdict(list)}
        self.outer_results = {"losses":defaultdict(list),
                    "accuracy":defaultdict(list)}
        self.test_results = {"losses":defaultdict(list),
                    "accuracy":defaultdict(list)}

        self.plotter = Plotter(self.exp_name)

    def _initialize_loaders(self, task, type_="train"):
        if type_ == "train":
            return self.train_datasets[task].get_dataloaders()
        elif type_ == "valid":
            return self.valid_datasets[task].get_dataloaders()
        else:
            return self.test_datasets[task].get_dataloaders()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def dump_results(self):
        with open(os.path.join(self.results_save_path, self.exp_name + '.txt'), 'w') as f:
            json.dump({"inner":self.inner_results,
                "outer": self.outer_results,
                "test": self.test_results}, f)

    def train(self, test_every=1):
        """Run episodes and perform outerloop updates """
        for epoch in range(self.n_epochs):
            self.outer_optimizer.zero_grad()
            for episode in range(self.num_episodes):
                print("---- Starting episode {} of epoch {} ----".format(episode, epoch))
                support_set, query_set = self.sample()
                self.train_episode(support_set, query_set, episode)

            if epoch % test_every == 0:
                test_loss, test_acc = self.evaluate_on_test_set(episode)
                print("Test performance: epoch {}, task {}, loss: {}, accuracy: {}".format(
                        epoch, episode, test_loss, test_acc))

            self.outer_optimizer.step()
            self.outer_lr_scheduler.step()

            self.dump_results()
            torch.save(self.outer_model.state_dict(), os.path.join(self.model_save_path, self.exp_name + '.pt'))
        self.plotter.plot()


    def forward(self, model, batch):
        logits = model(self._to_device(batch['input_ids']),
                    token_type_ids=self._to_device(batch['token_type_ids']),
                    attention_mask=self._to_device(batch['attention_mask']))
        return logits

    def _to_device(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.tensor(inp)
        return inp.to(self.device)


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
                    self.valid_loaders[task] = self._initialize_loaders(task, type_="valid")
                    b = next(self.valid_loaders[task][i], -1)
                label = b["labels"][0].item()
                query_task_batch[label] = b
            query_set[task] = query_task_batch
        return source_set, query_set

    def init_prototype_parameters(self, model, support_set, task):
        n_classes = self.task_classes[task]
        prototypes = self._to_device(torch.zeros((n_classes, 768)))
        class_samples = self._to_device(torch.zeros((n_classes,1)))
        for label in range(n_classes):
            batches = support_set[task][label]
            # Batch is either a list of dicts, or a single dict.
            if isinstance(batches, dict):
                batches = [batches]

            n_batches = len(batches)
            for batch in batches:
                class_samples[label,:] += batch['input_ids'].size(0)
                input_ids = self._to_device(batch['input_ids'])
                token_type_ids = self._to_device(batch['token_type_ids'])
                attention_mask = self._to_device(batch['attention_mask'])

                encoding = self.outer_model.encoder(input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)["last_hidden_state"][:,0,:]
                prototypes[label, :] = prototypes[label, :] + self._to_device(torch.sum(encoding, dim=0))

        model.gamma = prototypes / class_samples

    def _extract(self, batch):
        shape = [batch[class_]["input_ids"].shape[1] for class_ in batch.keys()]
        max_shape = max(shape)
        input_ids = torch.cat(tuple([self._pad(batch[class_]["input_ids"], max_shape) for class_ in batch.keys()]), dim=0)
        token_type_ids = torch.cat(tuple([self._pad(batch[class_]["token_type_ids"], max_shape) for class_ in batch.keys()]), dim=0)
        attention_mask = torch.cat(tuple([self._pad(batch[class_]["attention_mask"], max_shape) for class_ in batch.keys()]), dim=0)
        labels = torch.cat(tuple([batch[class_]["labels"] for class_ in batch.keys()]))

        shuffle_indices = torch.randperm(labels.shape[0])
        return {'input_ids': input_ids[shuffle_indices],
                'token_type_ids': token_type_ids[shuffle_indices],
                'attention_mask': attention_mask[shuffle_indices],
                'labels': labels[shuffle_indices]
                }

    def _pad(self, tensor, max_shape):
        tensor = torch.nn.functional.pad(tensor, (0, max_shape - tensor.shape[1]), mode='constant', value=PAD_ID).detach()
        return tensor


    def inner_loop(self, model, support_set, task):

        inner_loss = []
        inner_acc = []
        loss_func = self.loss_funcs[task]
        n_classes = self.task_classes[task]
        model.zero_grad()
        optimizer = AdamW(model.parameters(), lr=self.inner_lr, weight_decay=1e-4)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=)

        support_samples = self._extract(support_set[task])
        support_len = len(support_samples['labels'])
        batch_idx = np.arange(0, support_len, self.inner_batch_size)
        for start_idx in batch_idx:
            batch = {k:s[start_idx:start_idx+self.inner_batch_size] for k, s in support_samples.items()}
            labels = self._to_device(batch['labels'])
            optimizer.zero_grad()
            logits = self.forward(model, batch)

            loss = loss_func(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            optimizer.step()

    def get_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=1)
        return (predictions == labels).float().mean().item()


    def calc_validation_grads(self, model, query_set, task):
        loss_func = self.loss_funcs[task]
        n_classes = self.task_classes[task]

        batch = self._extract(query_set[task])
        labels = self._to_device(batch["labels"])
        logits = self.forward(model, batch)

        loss = loss_func(logits, labels)
        accuracy = self.get_accuracy(logits, labels)
        print("task: {}, query accuracy: {}, query loss: {}".format(task, accuracy, loss.item()))
        self.outer_results["losses"][task].append(loss.item())
        self.outer_results["accuracy"][task].append(accuracy)

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


    def evaluate_on_test_set(self, task):
        self.outer_model.eval()
        loss_func = self.loss_funcs[task]
        n_classes = self.task_classes[task]

        self.test_loaders[task] = self._initialize_loaders(task, type_="test")

        losses, accuracies = [], []
        support_set = {task:{c:list(self.test_loaders[task][c]) for c in range(n_classes)}}
        with torch.no_grad():
            self.init_prototype_parameters(self.outer_model, support_set, task)
            self.outer_model.init_phi(n_classes)
            for c in range(n_classes):
                for batch in support_set[task][c]:
                    labels = self._to_device(batch["labels"])
                    logits = self.forward(self.outer_model, batch)
                    losses.append(loss_func(logits, labels).item())
                    accuracies.append(self.get_accuracy(logits, labels))

        self.outer_model.gamma = None
        self.outer_model.phi = None
        self.outer_model.train()
        avg_loss = np.mean(losses)
        avg_acc = np.mean(accuracies)
        self.test_results["losses"][task].append(avg_loss)
        self.test_results["accuracy"][task].append(avg_acc)
        self.plotter.update({"loss" : avg_loss, "accuracy": avg_acc})
        return avg_loss, avg_acc

    def train_episode(self, support_set, query_set, task):
        "train inner model for 1 step, returns gradients of encoder on support set."
        n_classes = self.task_classes[task]
        loss_func = self.loss_funcs[task]

        # Step 2: Duplicate model
        inner_model = copy.deepcopy(self.outer_model)

        # Step 3: Init prototype vectors (for now just take n embedding vectors).
        print("---- Initializing prototype parameters ----")
        self.init_prototype_parameters(inner_model, support_set, task)

        # Step 4: Init output parameters (phi).
        inner_model.init_phi(n_classes)

        # Step 5: perform k inner loop steps.
        print("---- Performing inner loop updates ----")
        self.inner_loop(inner_model, support_set, task)

        # Step 6: Replace output parameters with trick.
        inner_model.replace_phi()

        # Step 7: Apply trained model on query set.
        print("---- Calculating gradients on query set ----")
        self.calc_validation_grads(inner_model, query_set, task)






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--freeze_bert", action="store_true",
                        help="Whether to freeze BERT parameters.")
    parser.add_argument("--epochs", type=int, default=50000,
                        help="Number of outerloop updates to run.")
    parser.add_argument("--outer_lr", type=float, default=1e-4,
                        help="learning rate for outer loop optimizer.")
    parser.add_argument("--inner_lr", type=float, default=1e-4,
                        help="learning rate for inner loop optimizer.")
    parser.add_argument("--support_k", type=int, default=10,
                        help="Number of support samples for each class.")
    parser.add_argument("--query_k", type=int, default=10,
                        help="Number of query samples for each class.")
    parser.add_argument("--model_save_path", type=str, default="saved_models/",
                        help="location to store saved model")
    parser.add_argument("--results_save_path", type=str, default="results/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--clip_value", type=float, default=2.0)
    parser.add_argument("--device", default=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    parser.add_argument("--inner_loop_batch_size", type=int, default=16,
                        help="batch size for the inner loop updates.")
    parser.add_argument("--exp_name", default='default', type=str, help="Model and results will be saved here")



    config = parser.parse_args().__dict__

    model = Classifier(config)

    para_train_support, para_train_query = ParaphraseDataset.read(path='data/msrp/', split='train', ratio=0.5)
    para_train_support_metaset = MetaDataset.Initialize(para_train_support, config["support_k"])
    para_train_query_metaset = MetaDataset.Initialize(para_train_query, config["query_k"])

    para_test = ParaphraseDataset.read(path='data/msrp/', split='test')
    para_test_metaset = MetaDataset.Initialize(para_test, config["support_k"], test=True)


    #mnli_train_support = MNLI.read(path='./data/multinli_1.0/', split='train', slice_=-1)
    #mnli_train_query = MNLI.read(path='./data/multinli_1.0/', split='dev_matched')

    #mnli_train_support_metaset = MetaDataset.Initialize(mnli_train_support, config["support_k"])
    #mnli_train_query_metaset = MetaDataset.Initialize(mnli_train_query, config["query_k"])



    meta_trainer = MetaTrainer(
                            model = model,
                            train_datasets = [para_train_support_metaset],
                            val_datasets = [para_train_query_metaset],
                            test_datasets = [para_test_metaset],
                            task_classes = {0:2},
                            epochs = config["epochs"],
                            inner_lr = config['inner_lr'],
                            outer_lr = config['outer_lr'],
                            inner_batch_size = config["inner_loop_batch_size"],
                            num_episodes = config["n_episodes"],
                            model_save_path = config["model_save_path"],
                            results_save_path = config["results_save_path"],
                            device = config["device"],
                            clip_value = config["clip_value"],
                            exp_name = config["exp_name"],
                            seed = config["seed"]
                            )

    meta_trainer.train(test_every=1)
