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
from transformers import BertTokenizer
from collections import defaultdict

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
        self.vocab = self.data.vocab
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
        premise = pad_sequence([item['premise'] for item in batch], padding_value=self.data.vocab.pad_id)
        hypothesis = pad_sequence([item['hypothesis'] for item in batch], padding_value=self.data.vocab.pad_id)
        label = [item['label'] for item in batch]
        return {'premise': premise, 'hypothesis': hypothesis, 'label':label}




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


#This is obsolete
class Vocabulary(object):
    def __init__(self, dim=300, pretrained=True):
        self.embed_dim = dim
        self.pretrained = pretrained
        self.vec_path = './'
        self.embedding = 'glove.840B.300d'

    def build_vectors(self):
        #initialize the randomly initiliazed word-vectors
        fname = os.path.join(self.vec_path, self.embedding + '_aligned.pt')
        if not os.path.exists(fname):
            vocab_size = len(self.vocab)
            std = 1/torch.sqrt(torch.tensor(self.embed_dim))
            self.vectors = torch.normal(0.0, std, (vocab_size, self.embed_dim)).float()
            self.align()
            torch.save(self.vectors, fname)
        else:
            self.vectors = torch.load(fname)

    def align(self):
        if not self.pretrained:
            pass
        else:
            with open(os.path.join(self.vec_path, self.embedding +'.txt'), mode='r', encoding="utf-8") as f:
                pbar = tqdm(f)
                for line in pbar:
                    pbar.set_description("Aligning the word vectors...")
                    values = line.strip().split(" ")
                    word, vec = values[0], values[1:]
                    word_id = self.vocab.get(word, None)
                    if word_id == None:
                        continue
                    else:
                        self.vectors[word_id] = torch.tensor(list(map(float, vec)), dtype=torch.float)


class Vocab(Vocabulary):
    def __init__(self):
        Vocabulary.__init__(self)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad_id = self.tokenizer._convert_token_to_id("[PAD]")

        self.vocab = {'[PAD]':self.pad_id, '[UNK]':1}
        self.count = {'[PAD]':1, '[UNK]':1}
        self.words = 2
    
    def Sentence(self, sentence):
        numericalized = []
        for token in self.tokenizer.tokenize(sentence):
            numericalized.append(self.Word(token))
        return numericalized
    
    def Word(self, token):
        if token not in self.vocab.keys():
            self.vocab[token] = self.words
            self.words+=1
            self.count[token] = 1
            return self.vocab[token]
        else:
            self.count[token] += 1
            return self.vocab[token]
    
    def filter(self, threshold=0):
        return {k:v for k, v in self.vocab.items() if self.count[k] > threshold or k in ['[PAD]', '[UNK]']}
    
    def __len__(self):
        return len(self.vocab)
    
    def embed(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        return [self.vocab[token] if token in self.vocab.keys() else self.vocab['<unk>'] for token in tokens]


class MNLI(Dataset):
    def __init__(self, data, vocab, labels):
        super(MNLI, self).__init__()
        self.data = data
        self.vocab = vocab
        self.labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

    def __getitem__(self, index):
        ret = self.data[index]
        return ret 

    def __len__(self):
        return len(self.data)
     
    @classmethod   
    def read(cls, vocab = None, path = './multinli_1.0/', split='train', slice_=-1):
        if vocab == None:
            vocab = Vocab()
            flag = False    #set Flag = False to indicate that the vocab was not provided
        else:
            vocab = vocab
            flag = True
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
                        'premise':MNLI.preprocess(premise, vocab, flag),
                        'hypothesis':MNLI.preprocess(hypothesis, vocab, flag)}
                        )
        return cls(data, vocab, labels)
    
    @staticmethod
    def preprocess(sentence, vocab=None, flag=None, label=False):
        if not flag and not label:
            return torch.LongTensor(vocab.Sentence(sentence.lower()))
        elif flag and not label:
            return torch.LongTensor(vocab.embed(sentence.lower()))
        else:
            labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
            return labels[sentence]
    
    def prepare_BERT_batch(self, batch, batch_size):
        """Prepare batch for BERT.
        Adds the special tokens to input, creates the token_type_ids and attention mask tensors.
        """
        batch = batch
        input_ids = []
        token_type_ids = []
        for i in range(batch_size):
            seq_1 = batch['premise'][:,i].tolist()
            seq_2 = batch['hypothesis'][:,i].tolist()
            input_ids.append(self.vocab.tokenizer.build_inputs_with_special_tokens(seq_1, seq_2))
            token_type_ids.append(self.vocab.tokenizer.create_token_type_ids_from_sequences(seq_1, seq_2))

        input_ids = torch.LongTensor(input_ids)
        attention_mask = (input_ids != self.vocab.pad_id).long()
        token_type_ids = torch.LongTensor(token_type_ids)
        labels = torch.LongTensor(batch['label'])
        return (input_ids, token_type_ids, attention_mask, labels)




if __name__ == "__main__":

    train = MNLI.read(path='./multinli_1.0/', split='train', slice_=1000)

    metadataset = MetaDataset.Initialize(train)


    loader = MetaLoader(metadataset).get_data_loader(metadataset.dataloaders()[0])
    
    print(next(loader))