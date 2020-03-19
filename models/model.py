import sys

sys.path.append('/home/mlspeech/gshalev/gal/IC_NLI')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
import torch
from torch import nn

class BiLSTM_withMaxPooling(nn.Module):
    device = None


    def __init__(self, lstm_input_size, lstm_hidden_zise, run_device):
        super(BiLSTM_withMaxPooling, self).__init__()
        self.device = run_device
        self.bilstm = nn.LSTM(lstm_input_size, lstm_hidden_zise, bidirectional=True)
        self.fc1 = nn.Linear(lstm_hidden_zise * 8, 512)
        self.fc2 = nn.Linear(512, 2)
        self.embedding = nn.Embedding(9490, 512)  # embedding layer


    def forward(self, sent1, sent2):
        u = self.encode(sent1)
        v = self.encode(sent2)

        representation = torch.cat((u, v, torch.abs(u - v), u * v), 1)

        out = self.fc1(representation)
        out = torch.tanh(out)
        out = self.fc2(out)
        return out


    def encode(self, sent):
        input, seq_len = sent
        input = input.to(self.device)
        input = self.embedding(input)
        if len(input.shape) == 3:
            input = input.permute(1, 0, 2)  # seq_len, batch, input_size
        else:
            input = input.unsqueeze(1)
        input = input.to(self.device)
        output = self.bilstm(input)[0]
        max_pooling = torch.max(output, 0)[0].to(self.device)
        return max_pooling

# model.py