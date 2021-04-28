import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from torchtext.data import Field, Iterator
from torchtext.datasets import MultiNLI
from transformers import BertTokenizer

class MNLIDataset():

    def __init__(self, config):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.batch_size = config['batch_size']
        self.pad_id = self.tokenizer._convert_token_to_id("[PAD]")

        self.text = Field(sequential=True, 
                        lower=True, 
                        tokenize=self.tokenizer.tokenize,
                        batch_first=True,
                        pad_token='[PAD]',
                        unk_token='[UNK]')
        self.labels = Field(sequential=False, is_target=True)

        self.train, self.dev, self.test = MultiNLI.splits(self.text, self.labels)

        self.text.build_vocab(self.train, self.dev, self.test)
        self.labels.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = Iterator.splits((self.train, self.dev, self.test), batch_size=config['batch_size'], device=config['device'])

    def get_BERT_iter(self, iter_):
        for batch in iter_:
            batch = self.prepare_BERT_batch(batch)
            yield batch

    def prepare_BERT_batch(self, batch):
        input_ids = []
        token_type_ids = []
        for i in range(self.batch_size):
            seq_1 = batch.premise[i].tolist()
            seq_2 = batch.hypothesis[i].tolist()
            input_ids.append(self.tokenizer.build_inputs_with_special_tokens(seq_1, seq_2))
            token_type_ids.append(self.tokenizer.create_token_type_ids_from_sequences(seq_1, seq_2))

        input_ids = torch.LongTensor(input_ids)
        attention_mask = (input_ids != self.pad_id).long()
        token_type_ids = torch.LongTensor(token_type_ids)
        labels = batch.label - 1
        return (input_ids, token_type_ids, attention_mask, labels)


if __name__ == "__main__":

    config = {
                "batch_size": 8,
                "device": "cpu"
            }

    dataset = MNLIDataset(config)
    tmp_iter = dataset.get_BERT_iter(dataset.train_iter)
    input_ids, token_type_ids, labels = next(tmp_iter)
    print(input_ids.size(), token_type_ids.size(), labels.size())
