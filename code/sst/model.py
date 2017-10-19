import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

class SSTSequenceEncoder(nn.Module):

    def __init__(self, feature_dim, hidden_dim, seq_length, batch_size,
                 num_proposals, num_layers=1, dropout=0):
        super(SSTSequenceEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_proposals = num_proposals
        self.num_layers = num_layers
        self.dropout = dropout

        self.GRU_cell = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, self.num_proposals)

    def init_hidden(self):
        return autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, input_var):

        #If input is two dimensional
        if len(input_var.size()) == 2:
            input_var = torch.unsqueeze(input_var, 0)

        self.seq_length = int(input_var.size()[1])

#        all_states = []
#        all_outputs = []
#        for i in range(self.seq_length):
#            out, hidden = self.GRU_cell(torch.unsqueeze(input_var[:,i,:], 0), self.hidden)
#            all_states.append(torch.squeeze(hidden))
#            all_outputs.append(torch.squeeze(out))
#            self.hidden = hidden
#
#        all_states = torch.stack(all_states)
#        all_outputs = torch.stack(all_outputs)

        hidden = self.init_hidden()
        all_outputs, all_states = self.GRU_cell(input_var, hidden)
        all_outputs = all_outputs.contiguous()
        all_outputs = F.sigmoid(self.fc(all_outputs.view(-1, self.hidden_dim)))
        all_outputs = all_outputs.view(self.batch_size, self.seq_length, self.num_proposals)
        all_outputs = torch.clamp(all_outputs, 0.001, 0.999)
        return all_outputs, all_states

class WeightedCrossEntropy(nn.Module):

    def __init__(self, w0, w1):
        super(WeightedCrossEntropy, self).__init__()
        self.w0 = w0
        self.w1 = w1

    def forward(self, target, predictions):
        assert target.size()[-1] == self.w0.size()[0]

        term1 = self.w1 * target
        term2 = self.w0 * (1.0 - target)

        loss = -(term1.view(-1) * torch.log(predictions.view(-1)) + (1.0 - term2.view(-1)) *
                torch.log(1.0 - predictions.view(-1)))

        return torch.mean(loss)
