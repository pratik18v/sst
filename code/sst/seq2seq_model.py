import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

USE_CUDA = True

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, batch_size=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = inputs.size(0)
        output, hidden = self.gru(inputs, hidden)
        output = self.fc(output.view(-1, self.hidden_size))
        output = output.view(seq_len, self.batch_size, self.output_size)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, encoder_output_size, n_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder_output_size = encoder_output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.hidden_linear = nn.Linear(encoder_output_size, output_size)
        self.gru = nn.GRU(hidden_size, output_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        input_seq = input_seq.unsqueeze(0)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(input_seq, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = F.sigmoid(self.out(concat_output))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b].view(-1), encoder_outputs[i, b].unsqueeze(0).view(-1))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

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

        loss = torch.mean(loss)

        return loss
