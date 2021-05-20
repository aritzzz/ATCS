from __future__ import division
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from torch.utils.data import IterableDataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import json
import pandas as pd
from tqdm import tqdm
import pickle
from transformers import BertTokenizer, DataCollatorWithPadding
from collections import defaultdict
import random



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
    def __init__(self, dataset, K, test, seed):
        self.data = dataset
        self.K = K
        self.rearrange(test)
        self.set_seed(seed)
    
    @classmethod
    def Initialize(cls, dataset, K=10, test=False, seed=42):
        #initialize the MetaDataset from dataset
        return cls(dataset, K, test, seed)
        
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # we want to rearrange the dataset so as to make sampling meta-datasets easy
    def rearrange(self, test):
        rearrange_data = defaultdict(lambda: [])
        for triple in self.data:
            label = triple['label']
            rearrange_data[label].append(triple)
        rearrange_data = {key: SingletonDataset(value) for key, value in rearrange_data.items()}
        if not test:
            self.meta_data = self.balance_data(rearrange_data)
        else:
            self.meta_data = rearrange_data
        self.labels = list(rearrange_data.keys())
    
    def get_dataloaders(self):
        #return as many dataloaders as the number of classes in the data
        return [self.sampler(DataLoader(dataset, batch_size=self.K, collate_fn=self.collater, shuffle=True)) for dataset in self.meta_data.values()]

    def collater(self, batch):
        premise = TOKENIZER.pad([item['input'] for item in batch], padding=True)
        label = [item['label'] for item in batch]
        return {'input': premise, 'label': label}
    
    def balance_data(self, data):
        lens = [len(v) for k,v in data.items()]
        max_len = max(lens)
        d = {}
        for k, v in data.items():
            difference = max_len - len(v)
            sample = [random.choice(v) for _ in range(difference)]
            v = v + sample
            d[k] = v
        return d
    
    def sampler(self, iter_):
        for batch in iter_:
            batch = self.data.prepare_BERT_batch(batch,self.K)
            yield batch



class MultiTaskLoader:

    def __init__(self, src_loader, aux_loader):
        self.src_loader = src_loader
        self.aux_loader = aux_loader

    def __iter__(self):
        self.src_iter = iter(self.src_loader)
        self.aux_iter = iter(self.aux_loader)
        return self

    def __next__(self):
        try:
            src_batch = next(self.src_iter)
        except:
            raise StopIteration
        
        try:
            aux_batch = next(self.aux_iter)
        except StopIteration:
            self.aux_iter = iter(self.aux_loader)
            aux_batch = next(self.aux_iter)

        return src_batch, aux_batch


# class MetaLoader(object):
#     def __init__(self, dataset):
#         self.dataset = dataset
        
#     def get_data_loader(self, loaders):
#         return [self.get_BERT_iter(loader) for loader in loaders]
    
#     def get_BERT_iter(self, iter_):
#         """Wrapper around standard iterator which prepares each batch for BERT."""
#         for batch in iter_:
#             batch = self.dataset.data.prepare_BERT_batch(batch,self.dataset.K)
#             yield batch


def collater(batch):
        premise = TOKENIZER.pad([item['input'] for item in batch], padding=True)
        label = [item['label'] for item in batch]
        return {'input': premise, 'label': label}

def collator2(batch):
    labels = [b['label'] for b in batch]
    inputs = TOKENIZER.pad([b['input'] for b in batch], padding=True)
    
    return {
            "input_ids":torch.LongTensor(inputs['input_ids']),
            "token_type_ids":torch.LongTensor(inputs['token_type_ids']),
            "attention_mask":torch.LongTensor(inputs['attention_mask']),
            "labels":torch.LongTensor(labels)
        }


class BaseDataset(Dataset):
    def __init__(self, data, labels):
        super(BaseDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        ret = self.data[index]
        return ret 

    def __len__(self):
        return len(self.data)

    @staticmethod
    def preprocess(sentence, labels=None, label=False):
        if not label:
            return TOKENIZER(sentence[0], sentence[1], add_special_tokens=True)
        else:
            return labels[sentence]
    
    def prepare_BERT_batch(self, batch, batch_size=8):
        """Prepare batch for BERT.
        Adds the special tokens to input, creates the token_type_ids and attention mask tensors.
        """
        input_ids = batch['input']['input_ids']
        token_type_ids = batch['input']['token_type_ids']
        attention_mask = batch['input']['attention_mask']

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask) #(input_ids != PAD_ID).long()
        token_type_ids = torch.LongTensor(token_type_ids)
        labels = torch.LongTensor(batch['label'])
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "labels": labels}

class MNLI(BaseDataset):
    def __init__(self, data, labels):
        super(BaseDataset, self).__init__()
        self.data = data
        self.labels = labels 
     
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
                        {'label':MNLI.preprocess(label_, labels=labels, label=True),
                        'input':MNLI.preprocess((premise, hypothesis))}
                        )
        return cls(data, labels)
    

class ParaphraseDataset(BaseDataset):
    def __init__(self, data):
        super(BaseDataset, self).__init__()
        self.data = data

    @classmethod
    def read(cls, path='./data/msrp/', split='train', slice_=-1, ratio=1, random_seed=42):
        np.random.seed(42)
        split_path = os.path.join(path, 'msr_paraphrase_' + split + '.txt')

        data = []
        with open(split_path, 'r') as f:
            lines = f.readlines()[1:slice_]
        pbar = lines
        print("Reading and preparing dataset...")
        for line in pbar:
           # pbar.set_description("Reading and Preparing dataset...")
            attributes = line.lower().replace("\n", "").split("\t")
            label_ = int(attributes[0])
            sent1 = attributes[-2]
            sent2 = attributes[-1]

            data.append(
                    {"label":label_, 
                    "input":MNLI.preprocess((sent1, sent2))}
                )
        if ratio < 1:
            np.random.shuffle(np.array(data))
            n_train = int(len(data) * ratio)
            train_data = data[:n_train]
            dev_data = data[n_train:]

            return cls(train_data), cls(dev_data)
        else:
            return cls(data)


class StanceDataset(BaseDataset):
    def __init__(self, data, labels):
        super(BaseDataset, self).__init__()
        self.data = data
        self.labels = labels
    

    @classmethod
    def read(cls, path='./claim_stance/', split='train', slice_=-1, ratio=1, random_seed=42):
        split_path = os.path.join(path, 'claim_stance_dataset_v1' + '.csv')
        df = pd.read_csv(split_path)[["split", "topicText", "claims.claimCorrectedText", "claims.stance"]]

        data = []
        labels = {'PRO': 1, 'CON': 0}
        pbar = df.iterrows()
        print("Reading and Preparing dataset...")
        for _, row in pbar:
           # pbar.set_description("Reading and Preparing dataset...")
            if row['split'] != split:
                continue
            sent1 = row['topicText']
            sent2 = row['claims.claimCorrectedText']
            label_ = row['claims.stance']
            data.append(
                        {'label':StanceDataset.preprocess(label_, labels=labels, label=True),
                        'input':StanceDataset.preprocess((sent1, sent2))}
                        )
        if ratio < 1:
            np.random.shuffle(np.array(data))
            n_train = int(len(data) * ratio)
            train_data = data[:n_train]
            dev_data = data[n_train:]

            return cls(train_data, labels), cls(dev_data, labels)
        else:
            return cls(data, labels)






if __name__ == "__main__":

    train = MNLI.read(path='./multinli_1.0/', split='train', slice_=1000)

    metadataset = MetaDataset.Initialize(train)


    loader = metadataset.get_dataloaders()[0]
    
    count = 0
    while True:
        b = next(loader, -1)
        print(b)
        print(count, ": Ok")
        if b == -1:
            print("Oops! Loader exhausted. reinitializing...")
            #reinitialize dataloader...
            loader = metadataset.get_dataloaders()[0]
        count+=1
        break


    #train = StanceDataset.read(path="data/Stance/", split='train', slice_=1000)

    #metadataset = MetaDataset.Initialize(train)


    #loader = MetaLoader(metadataset).get_data_loader(metadataset.dataloaders()[0])
    
    #print(next(loader))

    # train = ParaphraseDataset.read(path="data/msrp/", split="train", slice_=1000)

    # metadataset = MetaDataset.Initialize(train)

    # loader = MetaLoader(metadataset).get_data_loader(metadataset.dataloaders()[0])

    # print(next(loader))

