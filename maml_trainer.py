import copy
import torch
import torch.nn as nn
import json
import argparse
from tqdm import tqdm
from data_utils import *
from models import Classifier
from collections import defaultdict

class MetaTrainer(object):


    def __init__(self, model, train_datasets, val_datasets, test_datasets, task_classes, epochs,
            num_episodes, model_save_path, results_save_path, seed=42, device=torch.device("cpu")):
        self.set_seed(seed)
        self.outer_model = model.to(device)
        self.n_tasks = num_episodes
        self.n_epochs = epochs
        self.model_save_path = model_save_path
        self.results_save_path = results_save_path
        self.train_datasets = train_datasets
        self.valid_datasets = val_datasets
        self.test_datasets = test_datasets

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
        self.outer_optimizer = torch.optim.AdamW(self.outer_model.encoder.parameters(),
                                                weight_decay=1e-4)

        self.inner_results = {"losses":defaultdict(list),
                    "accuracy":defaultdict(list)}
        self.outer_results = {"losses":defaultdict(list),
                    "accuracy":defaultdict(list)}
        self.test_results = {"losses":defaultdict(list),
                    "accuracy":defaultdict(list)}

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
        with open(self.results_save_path, 'w') as f:
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

            self.dump_results()
            torch.save(self.outer_model.state_dict(), self.model_save_path)


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
        loss_func = self.loss_funcs[task]
        n_classes = self.task_classes[task]
        model.zero_grad()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-4)

        batch = self._extract(support_set[task])
        labels = self._to_device(batch['labels'])
        optimizer.zero_grad()
        logits = self.forward(model, batch)

        loss = loss_func(logits, labels)
        accuracy = self.get_accuracy(logits, labels)
        self.inner_results["losses"][task].append(loss.item())
        self.inner_results["accuracy"][task].append(accuracy)

        loss.backward()
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
    parser.add_argument("--outer_lr", type=float, default=1e-3,
                        help="learning rate for outer loop optimizer.")
    parser.add_argument("--inner_lr", type=float, default=0.1,
                        help="learning rate for inner loop optimizer.")
    parser.add_argument("--support_k", type=int, default=10,
                        help="Number of support samples for each class.")
    parser.add_argument("--query_k", type=int, default=10,
                        help="Number of query samples for each class.")
    parser.add_argument("--model_save_path", type=str, default="saved_models/model.pt",
                        help="location to store saved model")
    parser.add_argument("--results_save_path", type=str, default="results/results.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--device", default=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))


    config = parser.parse_args().__dict__

    model = Classifier(config)

    para_train_support, para_train_query = ParaphraseDataset.read(path='data/msrp/', split='train', ratio=0.5)
    para_train_support_metaset = MetaDataset.Initialize(para_train_support, config["support_k"])
    para_train_query_metaset = MetaDataset.Initialize(para_train_query, config["query_k"])

    para_test = ParaphraseDataset.read(path='data/msrp/', split='test')
    para_test_metaset = MetaDataset.Initialize(para_test, config["support_k"], test=True)


    meta_trainer = MetaTrainer(
                            model = model,
                            train_datasets = [para_train_support_metaset],
                            val_datasets = [para_train_query_metaset],
                            test_datasets = [para_test_metaset],
                            task_classes = {0:2},
                            epochs = config["epochs"],
                            num_episodes = config["n_episodes"],
                            model_save_path = config["model_save_path"],
                            results_save_path = config["results_save_path"],
                            device = config["device"],
                            seed = config["seed"]
                            )

    meta_trainer.train(test_every=100)

