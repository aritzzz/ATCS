import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class Phi(nn.Module):

    def __init__(self, W, b):
        super(Linear, self).__init__()
        self.weight = W
        self.bias = b

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class Classifier(nn.Module):
    """Simple classifier which consists of an encoder(BERT) and an MLP."""

    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.gamma = None
        self.phi = None

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
        return x #self.phi(x)

    def init_phi(self, n_classes):
        self.phi = nn.Linear(768, n_classes)
        self.phi.weight = nn.Parameter(2 * self.gamma)
        self.phi.bias = nn.Parameter(-torch.sum(self.gamma * self.gamma, dim=-1))

        # detaching initialization
        self.phi.weight.data = self.phi.weight.data.detach()
        self.phi.bias.data = self.phi.bias.data.detach()

    def replace_phi(self):
        
        """For evaluation of the inner model on the evaluation set we dont need to update
        these parameters. Changing them from nn.Parameter to regular tensors allows the gradient
        to flow through initialization (using nn.Parameter does not).
        """
        w_data = self.phi.weight.data
        b_data = self.phi.bias.data

        del self.phi.weight
        del self.phi.bias

        self.phi.weight = 2 * self.gamma + (w_data - 2 * self.gamma).detach()
        scalar_norm = -torch.sum(self.gamma * self.gamma, dim=-1)
        self.phi.bias = scalar_norm + (b_data - scalar_norm).detach()



if __name__ == "__main__":
    config = {"num_classes": 3, 'freeze_bert': True}
    model = Classifier(config)
    print(model)
