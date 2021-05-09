from __future__ import division
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import json
import pandas as pd
from tqdm import tqdm
import pickle
from transformers import BertTokenizer, DataCollatorWithPadding
from collections import defaultdict



TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
PAD_ID = TOKENIZER._convert_token_to_id("[PAD]")

class SingletonDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class MetaDataset(Dataset):
    def __init__(self, dataset, K):
        self.data = dataset
        self.K = K
        self.rearrange()
    
    @classmethod
    def Initialize(cls, dataset, K=10):
        #initialize the MetaDataset from dataset
        return cls(dataset, K)
        
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    # we want to rearrange the dataset so as to make sampling meta-datasets easy
    def rearrange(self):
        rearrange_data = defaultdict(lambda: [])
        for triple in self.data:
            label = triple['label']
            rearrange_data[label].append(triple)
        rearrange_data = {key: SingletonDataset(value) for key, value in rearrange_data.items()}
        self.meta_data = rearrange_data
        self.labels = list(rearrange_data.keys())
    
    def dataloaders(self):
        #return as many dataloaders as the number of classes in the data
        return [DataLoader(dataset, batch_size=self.K, collate_fn=self.collater) for dataset in self.meta_data.values()]

    def collater(self, batch):
        premise = TOKENIZER.pad([item['input'] for item in batch], padding=True)
        label = [item['label'] for item in batch]
        return {'input': premise, 'label': label}




class MetaLoader(object):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def get_data_loader(self, loader):
        return self.get_BERT_iter(loader)
    
    def get_BERT_iter(self, iter_):
        """Wrapper around standard iterator which prepares each batch for BERT."""
        for batch in iter_:
            batch = self.dataset.data.prepare_BERT_batch(batch,self.dataset.K)
            yield batch

class MNLI(Dataset):
    def __init__(self, data, labels):
        super(MNLI, self).__init__()
        self.data = data
        self.labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

    def __getitem__(self, index):
        ret = self.data[index]
        return ret 

    def __len__(self):
        return len(self.data)
     
    @classmethod   
    def read(cls, path = './multinli_1.0/', split='train', slice_=-1):
        labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        split_path = os.path.join(path, 'multinli_1.0_'+ split + '.jsonl')
        data = []
        with open(split_path, 'r') as f:
            lines = f.readlines()[:slice_]
        pbar = tqdm(lines)
        for line in pbar:
            pbar.set_description("Reading and Preparing dataset...")
            line = json.loads(line)
            label_ = line['gold_label']
            if label_ not in labels.keys():
                continue
            premise = line['sentence1']
            hypothesis = line['sentence2']
            data.append(
                        {'label':MNLI.preprocess(label_, label=True),
                        'input':MNLI.preprocess((premise, hypothesis))}
                        )
        return cls(data, labels)
    
    @staticmethod
    def preprocess(sentence, label=False):
        if not label:
            return TOKENIZER(sentence[0], sentence[1], add_special_tokens=True)
        else:
            labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
            return labels[sentence]
    
    def prepare_BERT_batch(self, batch, batch_size):
        """Prepare batch for BERT.
        Adds the special tokens to input, creates the token_type_ids and attention mask tensors.
        """
        batch = batch
        input_ids = batch['input']['input_ids']
        token_type_ids = batch['input']['token_type_ids']
        attention_mask = batch['input']['attention_mask']

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask) #(input_ids != PAD_ID).long()
        token_type_ids = torch.LongTensor(token_type_ids)
        labels = torch.LongTensor(batch['label'])
        return (input_ids, token_type_ids, attention_mask, labels)




if __name__ == "__main__":

    train = MNLI.read(path='./multinli_1.0/', split='train', slice_=1000)

    metadataset = MetaDataset.Initialize(train)


    loader = MetaLoader(metadataset).get_data_loader(metadataset.dataloaders()[0])
    
    print(next(loader))