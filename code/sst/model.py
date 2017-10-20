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

        #Defining layers
        self.GRU_cell = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.num_proposals)

    def init_hidden(self):
        return autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_var):

        #If input is two dimensional, batch size is 1
        if len(input_var.size()) == 2:
            input_var = torch.unsqueeze(input_var, 0)

        self.seq_length = int(input_var.size()[1])
        hidden = self.init_hidden()

        all_outputs, all_states = self.GRU_cell(input_var, hidden)

        all_outputs = all_outputs.contiguous()
        all_outputs = F.sigmoid(self.fc(all_outputs.view(-1, self.hidden_dim)))
        all_outputs = all_outputs.view(self.batch_size, self.seq_length, self.num_proposals)
        all_outputs = torch.clamp(all_outputs, 0.001, 0.999)

        return all_outputs, all_states

class StatePairwiseConcat(nn.Module):

    def __init__(self, feature_dim, hidden_dim, seq_length, batch_size,
                 num_proposals, num_layers=1, dropout=0):
        super(StatePairwiseConcat, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_proposals = num_proposals
        self.num_layers = num_layers
        self.dropout = dropout

        #Defining layers
        self.GRU_cell = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.ctx_linear = nn.Linear(2*self.hidden_dim*self.num_proposals, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.num_proposals)

    def init_hidden(self):
        return autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_var):

        #If input is two dimensional, batch size is 1
        if len(input_var.size()) == 2:
            input_var = torch.unsqueeze(input_var, 0)

        self.seq_length = int(input_var.size()[1])
        hidden = self.init_hidden()

        all_outputs = autograd.Variable(torch.zeros(self.batch_size, self.seq_length, self.hidden_dim))
        all_states = autograd.Variable(torch.zeros(self.seq_length, self.batch_size, self.hidden_dim))
        for i in range(self.seq_length):
            context_mat = autograd.Variable(torch.zeros(self.batch_size, 2*self.hidden_dim,
                self.num_proposals).cuda())
            output, state = self.GRU_cell(input_var[:,i,:].unsqueeze(1), hidden)
            all_states[i, :, :] = state.squeeze(0)
            for j in range(self.num_proposals):
                idx = i - j
                if idx < 0:
                    idx = 0
                context_mat[:,:,j] = torch.cat((state.squeeze(0), all_states[idx,:, :]), dim=1)
            ctx = F.sigmoid(self.ctx_linear(context_mat.view(self.batch_size, -1)))
            all_outputs[:, i, :] = output.squeeze(1) * ctx

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
