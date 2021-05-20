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
        return [self.optimizer]

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
        self.cosines = {
                    "total":[],
                    "parameters":{name:[] for name, _ in self.model.named_parameters()}
                }
        self.automatic_optimization = False

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

    def custom_backward(self, src_loss, aux_loss):
        self.model.zero_grad()
        
        src_grads = torch.autograd.grad(src_loss,
                                        inputs=self.model.parameters(),
                                        allow_unused=True)
        aux_grads = torch.autograd.grad(aux_loss,
                                        inputs=self.model.parameters(),
                                        allow_unused=True)

        packed_src, packed_aux = [], []
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if src_grads[i] is None and aux_grads[i] is None:
                continue
            elif src_grads[i] is not None and aux_grads[i] is None:
                grad = src_grads[i]
            elif aux_grads[i] is not None and src_grads[i] is None:
                grad = aux_grads[i]
            else:
                grad = (src_grads[i] + aux_grads[i]) / 2
                
                g1 = src_grads[i].flatten()
                g2 = aux_grads[i].flatten()
                self.cosines['parameters'][name].append(F.cosine_similarity(g1, g2, dim=0).item())
                packed_src.append(g1)
                packed_aux.append(g2)
            
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

        packed_src = torch.cat(packed_src)
        packed_aux = torch.cat(packed_aux)
        self.cosines['total'].append(F.cosine_similarity(packed_src, packed_aux, dim=0).item())


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        src_batch, aux_batch = batch
        src_labels = self._to_device(src_batch['labels'])
        aux_labels = self._to_device(aux_batch['labels'])

        src_logits = self.forward(src_batch, src=True)
        aux_logits = self.forward(aux_batch, src=False)

        src_loss = self.loss_func(src_logits, src_labels)
        aux_loss = self.loss_func(aux_logits, aux_labels)
        loss = (src_loss + aux_loss) / 2

        src_acc = self.calc_accuracy(src_logits, src_labels)
        aux_acc = self.calc_accuracy(aux_logits, aux_labels)

        opt.zero_grad()
        self.custom_backward(src_loss, aux_loss)
        opt.step()

        self.log('src_train_loss', src_loss)
        self.log('aux_train_loss', aux_loss)
        self.log('src_train_accuracy', src_acc, on_step=False, on_epoch=True)
        self.log('aux_train_accuracy', aux_acc, on_step=False, on_epoch=True)
        
        return loss

    def training_epoch_end(self, training_step_outputs):
        with open(self.config['cosine_file'], 'w') as f:
            json.dump(self.cosines, f)

    def evaluation_step(self, batch):
        src_batch, aux_batch = batch
        src_labels = self._to_device(src_batch['labels'])
        aux_labels = self._to_device(aux_batch['labels'])

        src_logits = self.forward(src_batch, src=True)
        aux_logits = self.forward(aux_batch, src=False)

        src_loss = self.loss_func(src_logits, src_labels)
        aux_loss = self.loss_func(aux_logits, aux_labels)

        src_acc = self.calc_accuracy(src_logits, src_labels)
        aux_acc = self.calc_accuracy(aux_logits, aux_labels)

        return (src_loss, src_acc), (aux_loss, aux_acc)

    def validation_step(self, batch, batch_idx):
        (src_loss, src_acc), (aux_loss, aux_acc) = self.evaluation_step(batch)

        self.log('src_val_loss', src_loss)
        self.log('src_val_acc', src_acc)
        self.log('aux_val_loss', aux_loss)
        self.log('aux_val_acc', aux_acc)

    def test_step(self, batch, batch_idx):
        (src_loss, src_acc), (aux_loss, aux_acc) = self.evaluation_step(batch)

        self.log('src_test_loss', src_loss)
        self.log('src_test_acc', src_acc)
        self.log('aux_test_loss', aux_loss)
        self.log('aux_test_acc', aux_acc)


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters())
        return opt



def train_model(loaders, config):
    """Creates a trainer module and fits it on training + evaluation step.
    Performs evaluation on validation and test set after training.
    """
    model_save_path = os.path.join(config['model_save_path'], config["save_name"])
    n_gpus = 1 if torch.cuda.is_available() else 0
    early_stop_callback = EarlyStopping(
                monitor='validation_loss',
                patience=10,
                verbose=False,
                mode='min'
            )

    trainer = pl.Trainer(default_root_dir = model_save_path,
                        checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode='min', monitor='validation_loss'),
                        callbacks = [early_stop_callback],
                        gpus = n_gpus,
                        max_epochs = config['max_epochs'],
                        progress_bar_refresh_rate=1)
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
                        checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode='min', monitor='src_val_loss'),
                        gpus = n_gpus,
                        callbacks = [early_stop_callback],
                        max_epochs = config['max_epochs'],
                        progress_bar_refresh_rate=1,
                        reload_dataloaders_every_epoch=True)
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    train_loader = iter(MultiTaskLoader(src_loaders[0], aux_loaders[0]))
    dev_loader = iter(MultiTaskLoader(src_loaders[1], aux_loaders[1]))
    test_loader = iter(MultiTaskLoader(src_loaders[2], aux_loaders[2]))

    if config["pretrained_model_path"] != "":
        model = MultitaskTrainer.load_from_checkpoint(config['pretrained_model_path'])
    else:
        pl.seed_everything(config["seed"])
        model = MultiTaskTrainer(config)
        trainer.fit(model, train_loader, dev_loader)

    validation_result = trainer.test(model, test_dataloaders=dev_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)

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
    parser.add_argument("--lr", type=float, default=0.001,
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

    # Directory Arguments
    parser.add_argument("--model_save_path", type=str, default="checkpoints",
                        help="Directory to store trained models.")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="Path to pretrained model.")
    parser.add_argument("--save_name", type=str, default="bert_mlp")
    parser.add_argument("--cosine_file", type=str, default="checkpoints/cosine.txt")
    parser.add_argument("--dataset_file", type=str, default="data/Stance/claim_stance_dataset_v1.csv")
    
    config = parser.parse_args().__dict__

    para_train, para_dev = ParaphraseDataset.read(path='data/msrp/', split='train', ratio=0.7)
    para_train_load = DataLoader(para_train, 
                                batch_size=config['batch_size'], 
                                shuffle=True,
                                collate_fn=collator2)
    para_dev_load = DataLoader(para_dev, 
                            batch_size=config['batch_size'], 
                            shuffle=True,
                            collate_fn=collator2)

    para_test = ParaphraseDataset.read(path='data/msrp/', split='test')
    para_test_load = DataLoader(para_test, 
                                batch_size=config['batch_size'], 
                                shuffle=True,
                                collate_fn=collator2)

    if config["multitask"]:
        stance_train, stance_dev = StanceDataset.read(path='data/Stance/', split='train', ratio=0.7)
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
