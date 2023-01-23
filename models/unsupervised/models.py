import torch
import torch.nn as nn
from torch.nn import Linear

from utils.utils import set_device

DEVICE = set_device()

class LstmModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 20, num_layers: int = 2, output_dim: int = 1):
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
        super(LstmModel, self).__init__()
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
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        """Forward pass of model network

        Inputs:
            input: pytorch tensor (batch, channels, sequence)
                batch of input data

        Outputs:
            out: pytorch tensor (batch, labels)
                batch of labels
        """
        # out, hidden = self.lstm(input)# (batch, channels, sequence) -> [sequence, batch, channels]
        sensors = inputs.shape[2]
        h_0 = torch.zeros(self.num_layers, sensors, self.hidden_dim).requires_grad_().to(DEVICE)
        c_0 = torch.zeros(self.num_layers, sensors, self.hidden_dim).requires_grad_().to(DEVICE)

        _, (h_n, _) = self.lstm(inputs.permute(0, 2, 1), (h_0, c_0))
        out = self.fc_block(h_n)
        return out


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class TimeSeriesTransformers(nn.Module):
    def __init__(
            self,
            n_encoder_inputs,
            n_decoder_inputs,
            channels=512,
            dropout=0.1,
            lr=1e-4,
    ):
        super().__init__()

        # self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.input_projection = Linear(n_encoder_inputs, channels)
        self.output_projection = Linear(n_decoder_inputs, channels)

        self.linear = Linear(channels, 4)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        # print(f"src before calc {src.shape}")
        src_start = self.input_projection(src).permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start

        return src

    def decode_trg(self, trg, memory):
        # print(f"trg before calc {trg.shape}")
        # print(f"memory before calc {memory.shape}")

        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        # print(f"trg apply decoder {trg.shape}")
        # print(f"memory apply decoder {memory.shape}")
        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def forward(self, x):
        src = x

        # src = x
        # print("---------- Encoder")
        src = self.encode_src(src)
        # print(f"src after calc {src.shape}")

        trg = x
        # print("---------- Decoder")
        out = self.decode_trg(trg=trg, memory=src)

        return out

    # def training_step(self, batch, batch_idx):
    #     src, trg_in, trg_out = batch
    #
    #     y_hat = self.forward((src, trg_in))
    #
    #     y_hat = y_hat.view(-1)
    #     y = trg_out.view(-1)
    #
    #     loss = smape_loss(y_hat, y)
    #
    #     # self.log("train_loss", loss)
    #
    #     return loss
    #
    # def validation_step(self, batch, batch_idx):
    #     src, trg_in, trg_out = batch
    #
    #     y_hat = self.forward((src, trg_in))
    #
    #     y_hat = y_hat.view(-1)
    #     y = trg_out.view(-1)
    #
    #     loss = smape_loss(y_hat, y)
    #
    #     # self.log("valid_loss", loss)
    #
    #     return loss
    #
    # def test_step(self, batch, batch_idx):
    #     src, trg_in, trg_out = batch
    #
    #     y_hat = self.forward((src, trg_in))
    #
    #     y_hat = y_hat.view(-1)
    #     y = trg_out.view(-1)
    #
    #     loss = smape_loss(y_hat, y)
    #
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


if __name__ == "__main__":
    # source = torch.rand(size=(2, 16, 9))
    source = torch.rand(size=(2, 16, 8))
    target_in = torch.rand(size=(2, 16, 8))
    target_out = torch.rand(size=(2, 16, 1))

    print(f"source input: {source}")
    print(f"source target_in: {target_in}")
    print(f"source target_out: {target_out}")

    ts = TimeSeriesTransformers(n_encoder_inputs=8, n_decoder_inputs=8)

    pred = ts.forward((source, target_in))

    # print(pred.size())

    # ts.training_step((source, target_in, target_out), batch_idx=1)
