import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import argparse
import os

from models import Classifier, MultiTaskClassifier
from data_utils import *
#from data import MNLIDataset, StanceDataset, ParaphraseDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

class BaselineTrainer(pl.LightningModule):
    """Trainer for the baseline experiments which trains BERT + MLP."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = Classifier(config)
        self.model.init_phi_normal(config['n_classes'])
        self.config = config
        self.loss_func = self.get_loss_func()

    def configure_optimizers(self):
        """Configures optimizer for pytorch lightning."""
        optimizer_dict = {
                    "sgd": optim.SGD,
                    "adam": optim.Adam
                }
        optimizer = optimizer_dict[self.config['optimizer']]
        self.optimizer = optimizer(self.model.parameters(), self.config['lr'])
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                        num_training_steps=config['n_steps'],
                                                        num_warmup_steps=int(0.10*config['n_steps']))
        return [self.optimizer], self.scheduler

    def get_loss_func(self):
        """Returns loss function specified in config."""
        loss_dict = {
                    "ce": nn.CrossEntropyLoss(),
                    "bce": nn.BCEWithLogitsLoss()
                }
        return loss_dict[self.config['loss']]

    def forward(self, batch):
        """Performs a forward pass on the batch."""
        logits = self.model(self._to_device(batch['input_ids']), 
                    token_type_ids=self._to_device(batch['token_type_ids']), 
                    attention_mask=self._to_device(batch['attention_mask']))
        return logits

    def _to_device(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.tensor(inp)
        return inp.to(self.config['device'])

    def calc_accuracy(self, logits, labels):
        """Utility function to calculate the accuracy given logits and labels. """
        if logits.size(-1) > 1:
            predictions = torch.argmax(logits, dim=1)
        else:
            predictions = (nn.functional.sigmoid(logits) > 0.5).float()
        return (predictions == labels).float().mean().item()

    def training_step(self, batch, batch_idx):
        """Performs a single training step and logs metrics."""
        labels = self._to_device(batch['labels'])
        logits = self.forward(batch)
        loss = self.loss_func(logits, labels)
        accuracy = self.calc_accuracy(logits, labels)

        sch = self.scheduler
        sch.step()

        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        return loss

    def evaluation_step(self, batch):
        """Performs a single evaluation step."""
        labels = self._to_device(batch['labels'])
        logits = self.forward(batch)
        loss = self.loss_func(logits, labels)
        accuracy = self.calc_accuracy(logits, labels)

        return loss, accuracy

    def validation_step(self, batch, batch_idx):
        """Performs an evaluation step and logs metrics for validation set. """
        loss, accuracy = self.evaluation_step(batch)

        self.log('validation_loss', loss)
        self.log('validation_accuracy', accuracy)

    def test_step(self, batch, batch_idx):
        """Performs an evaluation step and logs metrics for test set."""
        loss, accuracy = self.evaluation_step(batch)

        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)


class MultiTaskTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiTaskClassifier(config)
        self.config = config
        self.loss_func = nn.CrossEntropyLoss()
        self.cosines = {"total":[]}
        self.automatic_optimization = False
        self.lambda_ = self.config['lambda']
        self.grad_cum = config['grad_cum']
        self.task_grad_cums = {0: [0, None], 1: [0, None]}
        self.src_loss = 0
        self.aux_loss = 0

    def forward(self, batch, src):
        """Performs a forward pass on the batch."""
        logits = self.model(self._to_device(batch['input_ids']),
                    token_type_ids=self._to_device(batch['token_type_ids']),
                    attention_mask=self._to_device(batch['attention_mask']),
                    src=src)
        return logits

    def _to_device(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.tensor(inp)
        return inp.to(self.config['device'])                

    def calc_accuracy(self, logits, labels):
        """Utility function to calculate the accuracy given logits and labels. """
        if logits.size(-1) > 1:
            predictions = torch.argmax(logits, dim=1)
        else:
            predictions = (nn.functional.sigmoid(logits) > 0.5).float()
        return (predictions == labels).float().mean().item()

    def accumulate_grads(self, task):

        packed_grads = []
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if 'src' not in name and 'aux' not in name and not 'encoder.pooler' in name:
                packed_grads.append(param.grad.flatten())

        packed_grads = torch.cat(packed_grads)
        if self.task_grad_cums[task][1] is None:
            self.task_grad_cums[task][1] = packed_grads
            self.task_grad_cums[task][0] += 1
        else:
            self.task_grad_cums[task][1] += packed_grads
            self.task_grad_cums[task][0] += 1

    def calc_cosine_sim(self, task):
        # check if we should accumulate or skip.
        if self.task_grad_cums[task][0] < self.grad_cum:
            self.accumulate_grads(task)
        
        # check if we can calculate cosine similarity.
        if self.task_grad_cums[0][0] == self.grad_cum and self.task_grad_cums[1][0] == self.grad_cum:
            cosine_sim = F.cosine_similarity(self.task_grad_cums[0][1], 
                                            self.task_grad_cums[1][1],
                                            dim=0).item()
            self.cosines['total'].append(cosine_sim)
            
            # reset task_grads:
            self.task_grad_cums = self.task_grad_cums = {0: [0, None], 1: [0, None]}

    def calc_lambda(self, loss):
        self.lambda_ = (0.1 * self.src_loss - self.aux_loss) / loss

    def training_step(self, batch, batch_idx):
        task, samples = batch 
        opt = self.optimizers()
        labels = self._to_device(samples['labels'])

        src = bool(1 - task)
        logits = self.forward(samples, src=src)
        loss = self.loss_func(logits, labels)
        acc = self.calc_accuracy(logits, labels)
        
        if src:
            self.src_loss += loss.item()
        else:
            self.calc_lambda(loss.item())
            self.aux_loss += self.lambda_ * loss.item()
            loss = self.lambda_ * loss

        opt.zero_grad()
        loss.backward()
        self.calc_cosine_sim(task) 
        opt.step()

        sch = self.scheduler
        sch.step()

        if src:
            self.log('src_train_loss', loss)
            self.log('src_train_accuracy', acc)
        else:
            self.log('aux_train_loss', loss)
            self.log('aux_train_accuracy', acc)
        
        return loss

    def training_epoch_end(self, training_step_outputs):
        with open(self.config['cosine_file'], 'w') as f:
            json.dump(self.cosines, f)

    def evaluation_step(self, samples, task):
        labels = self._to_device(samples['labels'])

        src = bool(1 - task)
        logits = self.forward(samples, src=src)

        loss = self.loss_func(logits, labels)
        acc = self.calc_accuracy(logits, labels)

        return loss, acc

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, acc = self.evaluation_step(batch, dataloader_idx)

        if dataloader_idx == 0:
            self.log('src_val_loss', loss)
            self.log('src_val_acc', acc)
        else:
            self.log('aux_val_loss', loss)
            self.log('aux_val_acc', acc)

    def test_step(self, batch, batch_idx, dataloader_idx):
        loss, acc = self.evaluation_step(batch, dataloader_idx)

        if dataloader_idx == 0:
            self.log('src_test_loss', loss)
            self.log('src_test_acc', acc)
        else:
            self.log('aux_test_loss', loss)
            self.log('aux_test_acc', acc)


    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.model.parameters())
        self.scheduler = get_cosine_schedule_with_warmup(self.opt, 
                                                num_training_steps=config['n_steps'],
                                                num_warmup_steps=int(0.10*config['n_steps']))
        return [self.opt], [self.scheduler]


def train_model(loaders, config):
    """Creates a trainer module and fits it on training + evaluation step.
    Performs evaluation on validation and test set after training.
    """
    model_save_path = os.path.join(config['model_save_path'], config["save_name"])
    n_gpus = 1 if torch.cuda.is_available() else 0
    early_stop_callback = EarlyStopping(
                monitor='validation_accuracy',
                patience=10,
                verbose=False,
                mode='max'
            )

    trainer = pl.Trainer(default_root_dir = model_save_path,
                        checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode='max', monitor='validation_accuracy'),
                        gpus = n_gpus,
                        max_steps = config['max_steps'],
                        progress_bar_refresh_rate=1,
                        gradient_clip_val=config["clip_value"])
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    train_loader, dev_loader, test_loader = loaders

    if config['pretrained_model_path'] != "":
        model = BaselineTrainer.load_from_checkpoint(config['pretrained_model_path'])
    else:
        pl.seed_everything(config['seed'])
        model = BaselineTrainer(config)
        trainer.fit(model, train_loader, dev_loader)
        #model = BaselineTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    if dev_loader is not None:
        validation_result = trainer.test(model, test_dataloaders=dev_loader, verbose=False)
    else:
        validation_result = [0]
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)

    results = {
            "validation": validation_result[0],
            "test": test_result[0]
        }
    return results

def train_multitask(src_loaders, aux_loaders, config):

    model_save_path = os.path.join(config["model_save_path"], config["save_name"])
    n_gpus = 1 if torch.cuda.is_available() else 0

    early_stop_callback = EarlyStopping(
                monitor='src_val_loss',
                patience=10,
                verbose=False,
                mode='min'
            )

    trainer = pl.Trainer(default_root_dir = model_save_path,
                        checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode='max', monitor='src_val_acc'),
                        gpus = n_gpus,
                        callbacks=[early_stop_callback],
                        val_check_interval=20,
                        max_steps = config['max_steps'],
                        progress_bar_refresh_rate=0,
                        gradient_clip_val=config['clip_value'])
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    train_loader = iter(MultiTaskLoader(src_loaders[0], aux_loaders[0], seed=config["seed"]))
    #dev_loader = iter(MultiTaskLoader(src_loaders[1], aux_loaders[1], seed=config["seed"]))
    #test_loader = iter(MultiTaskLoader(src_loaders[2], aux_loaders[2], seed=config["seed"]))

    if config["pretrained_model_path"] != "":
        model = MultiTaskTrainer.load_from_checkpoint(config['pretrained_model_path'])
    else:
        pl.seed_everything(config["seed"])
        model = MultiTaskTrainer(config)
        trainer.fit(model, train_loader, src_loaders[1])

    validation_result = trainer.test(model, test_dataloaders=[src_loaders[1], aux_loaders[1]], verbose=False)
    test_result = trainer.test(model, test_dataloaders=[src_loaders[2], aux_loaders[2]], verbose=False)

    results = {
            "validation": validation_result[0],
            "test": test_result[0]
        }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument("--n_classes", type=int, default=3,
                        help="Number of classes in the dataset.")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="Whether to freeze bert parameters.")
    parser.add_argument("--n_src_classes", type=int, default=2)
    parser.add_argument("--n_aux_classes", type=int, default=2)

    # Training Arguments
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Which optimizer to use.")
    parser.add_argument("--loss", type=str, default="ce",
                        help="Which loss function to use.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate for training.")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Size of training batches.")
    parser.add_argument("--device", default=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    parser.add_argument("--dataset_name", type=str, default="mnli",
                        help="Name of the dataset to train on.")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of training data to use for trainin set, the rest will go to val set.")
    parser.add_argument("--multitask", action="store_true",
                        help="whether to perform multitask training.")
    parser.add_argument("--lambda", type=float, default=0.1,
                        help="Ratio of aux loss w.r.t. src loss")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="maximum number of training steps.")

    # Directory Arguments
    parser.add_argument("--model_save_path", type=str, default="checkpoints",
                        help="Directory to store trained models.")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="Path to pretrained model.")
    parser.add_argument("--save_name", type=str, default="bert_mlp")
    parser.add_argument("--cosine_file", type=str, default="checkpoints/cosine.txt")
    parser.add_argument("--dataset_file", type=str, default="data/Stance/claim_stance_dataset_v1.csv")
    parser.add_argument("--grad_cum", type=int, default=4)
    parser.add_argument("--clip_value", type=float, default=1.5)

    config = parser.parse_args().__dict__

    para_train = MNLI.read(path='data/multinli_1.0/', split='train')
    para_dev = MNLI.read(path='data/multinli_1.0/', split='dev_matched')
    para_train_load = DataLoader(para_train, 
                                batch_size=config['batch_size'], 
                                shuffle=True,
                                collate_fn=collator2)
    para_dev_load = DataLoader(para_dev, 
                            batch_size=config['batch_size'], 
                            collate_fn=collator2)

    para_test = MNLI.read(path='data/multinli_1.0/', split='dev_mismatched')
    para_test_load = DataLoader(para_test, 
                                batch_size=config['batch_size'], 
                                collate_fn=collator2)

    config['n_steps'] = len(para_train) * config['max_epochs']

    if config["multitask"]:
        stance_train, stance_dev = StanceDataset.read(path='data/Stance/', split='train', ratio=0.9)
        stance_train_load = DataLoader(stance_train,
                                    batch_size=config['batch_size'],
                                    shuffle=True,
                                    collate_fn=collator2)
        stance_dev_load = DataLoader(stance_dev,
                                    batch_size=config['batch_size'],
                                    shuffle=True,
                                    collate_fn=collator2)
        stance_test = StanceDataset.read(path='data/Stance/', split='test')
        stance_test_load = DataLoader(stance_test,
                                    batch_size=config['batch_size'],
                                    shuffle=True,
                                    collate_fn=collator2) 
        
        train_multitask([para_train_load, para_dev_load, para_test_load],
                    [stance_train_load, stance_dev_load, stance_test_load],
                    config)

    else:
        train_model([para_train_load, para_dev_load, para_test_load], config)
