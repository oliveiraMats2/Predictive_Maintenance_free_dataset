from datasets import DatasetUnsupervisedMafaulda, DatasetSinteticUnsupervised, DatasetSinteticUnsupervisedLSTM
from datasets import DatasetWileC, Dataset_UCI, DatasetProphetTest, DatasetProphetTransformer
from losses import smape_loss, soft_dtw
from models.unsupervised.models import TimeSeriesTransformers, LstmModel, LstmModelConv, TransAm
from tst.transformer import Transformer
from tst.loss import OZELoss
import torch

FACTORY_DICT = {
    "model": {
        "TimeSeriesTransformers": TimeSeriesTransformers,
        "LstmModel": LstmModel,
        "LstmModelConv": LstmModelConv,
        "Transformer": Transformer,
        "TransAm": TransAm
    },
    "dataset": {
        "DatasetWileC": DatasetWileC,
        "DatasetUCI": Dataset_UCI,
        "DatasetUnsupervisedMafaulda": DatasetUnsupervisedMafaulda,
        "DatasetSinteticUnsupervised": DatasetSinteticUnsupervised,
        "DatasetSinteticUnsupervisedLSTM": DatasetSinteticUnsupervisedLSTM,
        "DatasetProphetTest": DatasetProphetTest,
        "DatasetProphetTransformer": DatasetProphetTransformer
    },
    "optimizer": {
        "Adam": torch.optim.Adam
    },
    "loss": {
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
        "smape_loss": smape_loss,
        "MSELoss": torch.nn.MSELoss(),
        "HingeLoss": torch.nn.HingeEmbeddingLoss(),
        "KullbackLeibler": torch.nn.KLDivLoss(reduction='batchmean'),
        "soft_dtw": soft_dtw,
        "OZELoss": OZELoss(alpha=0.3)
    },
}
