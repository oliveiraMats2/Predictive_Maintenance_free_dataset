import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 20, num_layers: int = 2, output_dim: int = 2):
        """Pytorch vanilla LSTM model for time series classification

        Arguments:
            input_dim : int
                number of channels (sensor time sequences)
            hidden_dim : int
                hidden layer size
            num_layers : int
                number of layers in LSTM block
            output_dim : int
                number of classification labels
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.num_layers)

        self.fc_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        """Forward pass of model network

        Inputs:
            input: pytorch tensor (batch, channels, sequence)
                batch of input data

        Outputs:
            out: pytorch tensor (batch, labels)
                batch of labels
        """
        # out, hidden = self.lstm(input)# (batch, channels, sequence) -> [sequence, batch, channels]
        out, hidden = self.lstm(input.permute(1, 0, 2))
        out = self.fc_block(out[-1])
        out = self.classifier(out)
        return F.softmax(out)


class LSTMattn(nn.Module):
    """
    https://github.com/prakashpandey9/Text-Classification-Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation

        Arguments:
            input_dim : int
                number of channels (sensor time sequences)
            hidden_dim : int
                hidden layer size
            num_layers : int
                number of layers in LSTM block
            output_dim : int
                number of classification labels
    """

    def __init__(self, input_dim, hidden_dim, num_layers=10, output_dim=2):
        super(LSTMattn, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.8)
        self.fc_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    def attention(self, lstm_output, hidden):
        """Luong attention model for sequence classification

        Inputs:
            lstm_output: pytorch tensor (sequence, batch, hidden)
                output of LSTM
            hidden: pytorch tensor (batch, hidden)
                hidden state of LSTM

        Outputs:
            output: pytorch tensor (batch, hidden)
                hidden state with applied attention
        """
        hidden = hidden.squeeze(0)
        lstm_output = lstm_output.permute(1, 0, 2)

        scores = torch.bmm(lstm_output, hidden.unsqueeze(2))
        attn_weights = F.softmax(scores, 1)  # eq.7
        context = torch.bmm(lstm_output.transpose(1, 2), attn_weights).squeeze(2)

        concat_input = torch.cat((hidden, context), 1)
        output = torch.tanh(self.concat(concat_input))  # eq. 5

        return output

    def forward(self, input):
        """Forward pass of model network

        Inputs:
            input: pytorch tensor (batch, channels, sequence)
                batch of input data

        Outputs:
            out: pytorch tensor (batch, labels)
                batch of labels
        """
        input = input.permute(1, 0, 2)
        lstm_out, (h, c) = self.lstm(input)
        out = self.attention(lstm_out, h[-1])
        out = self.classifier(out)

        return F.softmax(out)
