import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import argparse
import os

from models import Classifier
from data import MNLIDataset, StanceDataset, ParaphraseDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

class BaselineTrainer(pl.LightningModule):
    """Trainer for the baseline experiments which trains BERT + MLP."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = Classifier(config)
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
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return logits

    def calc_accuracy(self, logits, labels):
        """Utility function to calculate the accuracy given logits and labels. """
        if logits.size(-1) > 1:
            predictions = torch.argmax(logits, dim=1)
        else:
            predictions = (nn.functional.sigmoid(logits) > 0.5).float()
        return (predictions == labels).float().mean().item()

    def training_step(self, batch, batch_idx):
        """Performs a single training step and logs metrics."""
        labels = batch[-1]
        logits = self.forward(batch)
        loss = self.loss_func(logits, labels)
        accuracy = self.calc_accuracy(logits, labels)

        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        return loss

    def evaluation_step(self, batch):
        """Performs a single evaluation step."""
        labels = batch[-1]
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

def get_iters(dataset, config):
    """Get the iterators for the different datasets."""
    if config["dataset_name"] == "mnli":
        train_iter = dataset.get_BERT_iter(dataset.train_iter)
        val_iter = dataset.get_BERT_iter(dataset.dev_iter)
        test_iter = dataset.get_BERT_iter(dataset.test_iter)
        return train_iter, val_iter, test_iter
    else:
        train_iter = DataLoader(dataset.train_set, config["batch_size"])
        val_iter = DataLoader(dataset.val_set, config["batch_size"])
        test_iter = DataLoader(dataset.test_set, config["batch_size"])
        return train_iter, val_iter, test_iter


def train_model(dataset, config):
    """Creates a trainer module and fits it on training + evaluation step.
    Performs evaluation on validation and test set after training.
    """
    model_save_path = os.path.join(config['model_save_path'], config["save_name"])
    n_gpus = 1 if torch.cuda.is_available() else 0

    trainer = pl.Trainer(default_root_dir = model_save_path,
                        checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode='max', monitor='validation_accuracy'),
                        gpus = n_gpus,
                        max_epochs = config['max_epochs'],
                        progress_bar_refresh_rate=1)
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    train_iter, val_iter, test_iter = get_iters(dataset, config)

    if config['pretrained_model_path'] != "":
        model = BaselineTrainer.load_from_checkpoint(config['pretrained_model_path'])
    else:
        pl.seed_everything(config['seed'])
        model = BaselineTrainer(config)
        trainer.fit(model, train_iter, val_iter)
        #model = BaselineTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    if val_iter is not None:
        validation_result = trainer.test(model, test_dataloaders=val_iter, verbose=False)
    else:
        validation_result = [0]
    test_result = trainer.test(model, test_dataloaders=test_iter, verbose=False)

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

    # Directory Arguments
    parser.add_argument("--model_save_path", type=str, default="checkpoints",
                        help="Directory to store trained models.")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="Path to pretrained model.")
    parser.add_argument("--save_name", type=str, default="bert_mlp")
    parser.add_argument("--dataset_file", type=str, default="data/Stance/claim_stance_dataset_v1.csv")
    
    config = parser.parse_args().__dict__

    if config["dataset_name"] == "mnli":
        dataset = MNLIDataset(config)
    elif config["dataset_name"] == "stance":
        dataset = StanceDataset(config["dataset_file"], config)
    elif config["dataset_name"] == "paraphrase":
        dataset = ParaphraseDataset("data/msrp/msr_paraphrase_train.txt", "data/msrp/msr_paraphrase_test.txt", config)
    
    train_model(dataset, config)
