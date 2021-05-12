import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext.data import Field, Iterator
from torchtext.datasets import MultiNLI
from transformers import BertTokenizer


class MNLIDataset():
  """Class which loads the mnli data and creates the iterators."""

  def __init__(self, batch_size):
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.pad_id = self.tokenizer._convert_token_to_id("[PAD]")
    self.batch_size = batch_size
    # Objects in which the data will be stored.
    self.text = Field(sequential=True,
                      lower=True,
                      tokenize=self.tokenizer.tokenize,
                      batch_first=True,
                      pad_token='[PAD]',
                      unk_token='[UNK]')
    self.labels = Field(sequential=False, is_target=True)

    self.train, self.dev, self.test = MultiNLI.splits(self.text, self.labels)

    # Builds vocabulary for the data.
    self.text.build_vocab(self.train, self.dev, self.test)
    self.labels.build_vocab(self.train)
    
    self.train_size = len(self.train)
    self.val_size = len(self.dev)
    self.test_size = len(self.test)
    self.name = 'mnli'
    # Standard torchtext iterators, these do not return input suitable for BERT.
    self.train_iter, self.dev_iter, self.test_iter = Iterator.splits(
        (self.train, self.dev, self.test),
        batch_size=self.batch_size,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))  

  def _cache_splits(self):
    self.text = Field(sequential=True,
                      lower=True,
                      tokenize=self.tokenizer.tokenize,
                      batch_first=True,
                      pad_token='[PAD]',
                      unk_token='[UNK]')
    self.labels = Field(sequential=False, is_target=True)

    self.train, self.dev, self.test = MultiNLI.splits(self.text, self.labels)
    
    with open('../.data/multinli_1.0/train.json', 'w+') as f:
      for example in [vars(t) for t in self.train]:
        json.dump(example, f)
        f.write('\n')
        
    with open('../.data/multinli_1.0/val.json', 'w+') as f:
      for example in [vars(t) for t in self.dev]:
        json.dump(example, f)
        f.write('\n')
    with open('../.data/multinli_1.0/test.json', 'w+') as f:
      for example in [vars(t) for t in self.test]:
        json.dump(example, f)
        f.write('\n')
  
  def _load_splits(self):
    self.text = Field(sequential=True,
                      lower=True,
                      batch_first=True,
                      pad_token='[PAD]',
                      unk_token='[UNK]')
    self.labels = Field(sequential=False, is_target=True)
    fields = {'text': ('text', self.text), 'label': ('label', self.labels)}
    self.train, self.val, self.test = TabularDataset.splits(
                                              path = '../.data/multinli_1.0/',
                                              train = 'train.json',
                                              test = 'test.json',
                                              validation = 'val.json',
                                              format = 'json',
                                              fields = fields)
    
  def get_iter(self, split='train'):
    """Wrapper around standard iterator which prepares each batch for BERT."""
    if split == 'train':
      iter_ = self.train_iter
    elif split == 'val':
      iter_ = self.dev_iter
    else:
      _iter = self.test_iter
    for batch in iter_:
      batch = self._prepare_BERT_batch(batch)
      yield batch

  def _prepare_BERT_batch(self, batch):
    """Prepare batch for BERT.
    Adds the special tokens to input, creates the token_type_ids and attention mask tensors.
    """
    input_ids = []
    token_type_ids = []
    for i in range(self.batch_size):
      seq_1 = batch.premise[i].tolist()
      seq_2 = batch.hypothesis[i].tolist()
      input_ids.append(
          self.tokenizer.build_inputs_with_special_tokens(seq_1, seq_2))
      token_type_ids.append(
          self.tokenizer.create_token_type_ids_from_sequences(seq_1, seq_2))

    input_ids = torch.LongTensor(input_ids)
    attention_mask = (input_ids != self.pad_id).long()
    token_type_ids = torch.LongTensor(token_type_ids)
    labels = batch.label - 1
    return (input_ids, token_type_ids, attention_mask, labels)


class SplitDataset(Dataset):
  """Dataset object which takes data in the form of a dictionary (BertTokenizer output):
        - input_ids: list[list], tokenized input ids.
        - attention mask: list[list], attention mask for input.
        - token_type_ids: list[list], token type ids for the input_ids.
        - labels: list, labels.
    """

  def __init__(self, data):
    super().__init__()
    self.data = data

  def __len__(self):
    return len(self.data['input_ids'])

  def __getitem__(self, idx):
    input_ids = self.data["input_ids"][idx]
    attention_mask = self.data["attention_mask"][idx]
    label = self.data['labels'][idx]
    token_type_ids = self.data["token_type_ids"][idx]

    return (torch.LongTensor(input_ids), torch.LongTensor(token_type_ids),
            torch.LongTensor(attention_mask), torch.Tensor([label]))

  @classmethod
  def split_data(cls, data, ratio, seed):
    """Splits a data dictionary following a ratio. 
        Useful when training data needs to be split into validation and training data. """
    np.random.seed(seed)
    n_train = int(len(data["input_ids"]) * ratio)

    idx = np.arange(len(data["input_ids"]))
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_data = {k: np.array(data[k])[train_idx] for k in data.keys()}
    val_data = {k: np.array(data[k])[val_idx] for k in data.keys()}
    return train_data, val_data


class StanceDataset():
  """Class which loads stance data into train, validation, test splits.
    Each split is a SplitDataset object.
    """

  def __init__(self,
               batch_size,
               csv_path=".data/stance/claim_stance_dataset_v1.csv",
               train_split=0.7,
               seed=42):
    self.batch_size = batch_size
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train, test = self._create_data(csv_path)
    train, val = SplitDataset.split_data(train, train_split, seed)
    self.train_set = SplitDataset(train)
    self.val_set = SplitDataset(val)
    self.test_set = SplitDataset(test)

    self.train_size = len(self.train_set)
    self.val_size = len(self.val_set)
    self.test_size = len(self.test_set)
    self.name = 'stance'
    
  def get_iter(self, split='train'):
    if split == 'train':
      return DataLoader(self.train_set, batch_size=self.batch_size)
    elif split == 'val':
      return DataLoader(self.val_set, batch_size=self.batch_size)
    else:
      return DataLoader(self.test_set, batch_size=self.batch_size)

  def _create_data(self, csv_path):
    df = pd.read_csv(csv_path)[[
        "split", "topicText", "claims.claimCorrectedText", "claims.stance"
    ]]

    def _create_label(row):
      if row['claims.stance'] == "PRO":
        return 1
      else:
        return 0

    df['label'] = df.apply(lambda row: _create_label(row), axis=1)
    train_df, test_df = self._create_splits(df)
    train_data = self._prepare_input(train_df)
    train_data["labels"] = train_df["label"].values
    test_data = self._prepare_input(test_df)
    test_data["labels"] = test_df["label"].values
    return train_data, test_data

  def _create_splits(self, df):
    train_df = df.loc[df["split"] == "train"]
    test_df = df.loc[df["split"] == "test"]
    return train_df, test_df

  def _prepare_input(self, df):
    tokenized = self.tokenizer(list(df["topicText"]),
                               list(df["claims.claimCorrectedText"]),
                               padding=True)
    return tokenized


class ParaphraseDataset():
  """Class which loads Paraphrase data into train, validation and test splits.
    Each split is a SplitData object.
    """

  def __init__(self,
               batch_size,
               train_file=".data/msrp/msr_paraphrase_train.txt",
               test_file=".data/msrp/msr_paraphrase_test.txt",
               train_split=0.7,
               seed=42):
    self.batch_size = batch_size
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_set = self._prepare_data_from_txt(train_file)
    test_set = self._prepare_data_from_txt(test_file)
    train_set, val_set = SplitDataset.split_data(train_set, train_split, seed)

    self.train_set = SplitDataset(train_set)
    self.val_set = SplitDataset(val_set)
    self.test_set = SplitDataset(test_set)

    self.train_size = len(self.train_set)
    self.val_size = len(self.val_set)
    self.test_size = len(self.test_set)
    self.name = 'msrp'
    
  def get_iter(self, split='train'):
    if split == 'train':
      return DataLoader(self.train_set, batch_size=self.batch_size)
    elif split == 'val':
      return DataLoader(self.val_set, batch_size=self.batch_size)
    else:
      return DataLoader(self.test_set, batch_size=self.batch_size)

  def _prepare_data_from_txt(self, txt_file):
    s1, s2, targets = self._read_txt_file(txt_file)
    tokenized = self.tokenizer(s1, s2, padding=True)
    tokenized["labels"] = targets
    return tokenized

  def _read_txt_file(self, txt_file):
    s1, s2, targets = [], [], []
    with open(txt_file, "r") as handle:
      for line in handle.readlines()[1:]:
        features = line.lower().replace("\n", "").split("\t")
        targets.append(int(features[0]))
        s1.append(features[3])
        s2.append(features[4])

    return s1, s2, targets


if __name__ == "__main__":

  batch_size = 4
  dataset = 'msrp'

  if dataset == 'msrp':
    dataset = ParaphraseDataset()
    dataloader = DataLoader(dataset.val_set, batch_size)
    for batch in dataloader:
      print(batch[0].size(), batch[1].size(), batch[2].size(), batch[3].size())
  elif dataset == 'stance':
    dataset = StanceDataset()
    dataloader = DataLoader(dataset.val_set, batch_size)
    for batch in dataloader:
      print(batch[0].size(), batch[1].size(), batch[2].size(), batch[3].size())
  else:
    dataset = MNLIDataset(batch_size)
    tmp_iter = dataset.get_BERT_iter(dataset.train_iter)
    input_ids, token_type_ids, atten, labels = next(tmp_iter)
    print(input_ids.size(), token_type_ids.size(), atten.size(), labels.size())
