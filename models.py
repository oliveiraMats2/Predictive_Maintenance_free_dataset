import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, num_layers=2, output_dim=1):
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

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

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
        out, hidden = self.lstm(input.permute(2, 0, 1))  # (batch, channels, sequence) -> [sequence, batch, channels]
        out = self.fc_block(out[-1])
        out = self.classifier(out)
        return out
