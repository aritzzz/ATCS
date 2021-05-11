import torch.nn as nn
from transformers import BertModel


class Classifier(nn.Module):
    """Simple classifier which consists of an encoder(BERT) and an MLP."""

    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        # self.mlp = nn.Linear(768, config['n_classes'])

        if config['freeze_bert']:
            self.freeze_BERT()

    def freeze_BERT(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x, token_type_ids=None, attention_mask=None):
        # Take embedding from [CLS] token
        x = self.encoder(x, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask)["last_hidden_state"][:,0,:]
        return x #self.mlp(x)

    def init_mlp(self):
        """Possibly useful for the PROTO-MAML implementation, will remove if not."""
        raise NotImplementedError



if __name__ == "__main__":
    config = {"num_classes": 3, 'freeze_bert': True}
    model = Classifier(config)
    print(model)
