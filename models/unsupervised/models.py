import positional_encoder as pe
import torch
from torch import nn, Tensor
from torch.nn import Linear

from utils.utils import set_device

DEVICE = set_device()


class LstmModel(nn.Module):
    def __init__(self, num_sensors: int, hidden_units: int = 20, num_layers: int = 1, output_dim: int = 1):
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_dim = output_dim

        super(LstmModel, self).__init__()

        self.lstm_0 = nn.LSTM(input_size=self.num_sensors,
                              hidden_size=self.hidden_units,
                              batch_first=True,  # <<< very important
                              num_layers=self.num_layers)
        #
        # self.lstm_1 = nn.LSTM(input_size=self.num_sensors,
        #                       hidden_size=self.hidden_units,
        #                       batch_first=True,  # <<< very important
        #                       num_layers=self.num_layers)
        #
        # self.lstm_2 = nn.LSTM(input_size=self.num_sensors,
        #                       hidden_size=self.hidden_units,
        #                       batch_first=True,  # <<< very important
        #                       num_layers=self.num_layers)
        #
        # self.lstm_3 = nn.LSTM(input_size=self.num_sensors,
        #                       hidden_size=self.hidden_units,
        #                       batch_first=True,  # <<< very important
        #                       num_layers=self.num_layers)

        self.fc = nn.Linear(self.hidden_units, self.output_dim)

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
        batch_size = inputs.shape[0]
        h_0_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        c_0_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)

        # h_0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        # c_0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        #
        # h_0_2 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        # c_0_2 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        #
        # h_0_3 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        # c_0_3 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)

        sensor_0 = inputs[..., 0].unsqueeze(-1)
        # sensor_1 = inputs[..., 1].unsqueeze(-1)
        # sensor_2 = inputs[..., 2].unsqueeze(-1)
        # sensor_3 = inputs[..., 3].unsqueeze(-1)

        output_0, (h_n_0, _) = self.lstm_0(sensor_0, (h_0_0, c_0_0))
        # output_1, (h_n_1, _) = self.lstm_1(sensor_1, (h_0_1, c_0_1))
        # output_2, (h_n_2, _) = self.lstm_2(sensor_2, (h_0_2, c_0_2))
        # output_3, (h_n_3, _) = self.lstm_3(sensor_3, (h_0_3, c_0_3))

        out_embbeding_0 = self.fc(h_n_0[0])
        # out_embbeding_1 = self.fc(h_n_1[0])
        # out_embbeding_2 = self.fc(h_n_2[0])
        # out_embbeding_3 = self.fc(h_n_3[0])

        # out_sensors = torch.cat([out_embbeding_0,
        #                          out_embbeding_1,
        #                          out_embbeding_2,
        #                          out_embbeding_3], axis=1)

        return out_embbeding_0


class LstmModelConv(nn.Module):
    def __init__(self, num_sensors: int, hidden_units: int = 20, hidden_dim: int = 640, num_layers: int = 1,
                 output_dim: int = 1):
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        super(LstmModelConv, self).__init__()

        self.lstm_0 = nn.LSTM(input_size=self.num_sensors,
                              hidden_size=self.hidden_units,
                              batch_first=True,  # <<< very important
                              num_layers=self.num_layers)

        self.lstm_1 = nn.LSTM(input_size=self.num_sensors,
                              hidden_size=self.hidden_units,
                              batch_first=True,  # <<< very important
                              num_layers=self.num_layers)

        self.lstm_2 = nn.LSTM(input_size=self.num_sensors,
                              hidden_size=self.hidden_units,
                              batch_first=True,  # <<< very important
                              num_layers=self.num_layers)

        self.lstm_3 = nn.LSTM(input_size=self.num_sensors,
                              hidden_size=self.hidden_units,
                              batch_first=True,  # <<< very important
                              num_layers=self.num_layers)

        self.fc = nn.Linear(self.hidden_dim, self.output_dim + 2) # somar mais dois para casar a conta no final.

        self.conv_1 = nn.Conv2d(1, 32, (3, 3), padding=0)

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
        batch_size = inputs.shape[0]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)

        sensor_0 = inputs[..., 0].unsqueeze(-1)
        sensor_1 = inputs[..., 1].unsqueeze(-1)
        sensor_2 = inputs[..., 2].unsqueeze(-1)
        sensor_3 = inputs[..., 3].unsqueeze(-1)

        output_0, (h_n_0, _) = self.lstm_0(sensor_0, (h_0, c_0))
        output_1, (h_n_1, _) = self.lstm_1(sensor_1, (h_0, c_0))
        output_2, (h_n_2, _) = self.lstm_2(sensor_2, (h_0, c_0))
        output_3, (h_n_3, _) = self.lstm_3(sensor_3, (h_0, c_0))

        out_embbeding_0 = self.fc(h_n_0[-1])
        out_embbeding_1 = self.fc(h_n_1[-1])
        out_embbeding_2 = self.fc(h_n_2[-1])
        out_embbeding_3 = self.fc(h_n_3[-1])

        vector_ones = torch.ones(batch_size, out_embbeding_0.shape[1], 1).to(DEVICE)

        out_sensors = torch.cat([out_embbeding_0.unsqueeze(-1),
                                 out_embbeding_1.unsqueeze(-1),
                                 out_embbeding_2.unsqueeze(-1),
                                 out_embbeding_3.unsqueeze(-1),
                                 vector_ones,# in the end, I need vector 8x4
                                 vector_ones], axis=2)


        #https://madebyollin.github.io/convnet-calculator/
        out_sensors = self.conv_1(out_sensors.unsqueeze(1))# put channel = 1

        return out_sensors.mean(axis=1).mean(axis=1)


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask

class TimeSeriesTransformer(nn.Module):

    def __init__(self,
                 input_size: int,
                 dec_seq_len: int,
                 batch_first: bool,
                 out_seq_len: int = 58,
                 dim_val: int = 512,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_decoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 1
                 ):
        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer
                                     of the decoder
            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__()

        self.dec_seq_len = dec_seq_len

        # print("input_size is: {}".format(input_size))
        # print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

        # Create positional encoder
        self.positional_encoding_layer = pe.PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
        )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
        )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None,
                tgt_mask: Tensor = None) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]

        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """

        # print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        # print("From model.forward(): tgt size = {}".format(tgt.size()))

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(
            src)  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(
            src)  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder(  # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
        )
        # print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(
            tgt)  # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        # print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

        # if src_mask is not None:
        # print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        # if tgt_mask is not None:
        # print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output)  # shape [batch_size, target seq len]
        # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))

        return decoder_output


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

        self.input_projection = Linear(1, channels)# 1 de sensores
        self.output_projection = Linear(1, channels)

        self.linear = Linear(channels, 1)

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
