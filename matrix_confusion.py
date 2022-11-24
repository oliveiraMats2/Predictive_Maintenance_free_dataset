import matplotlib.pyplot as plt
import itertools
import numpy as np
from models import LSTM, LSTMattn
from utils.utils import *
from utils.read_dataset import read_h5
from tqdm import trange
from DataLoaders.data_loaders import DatasetWileC


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          dir_artifacts='artifacts/matrix_confusion.png',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=0)
    # plt.ylim(-0.5, len(names) - 0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(dir_artifacts)


if __name__ == '__main__':
    configs = read_yaml('configs/config_model.yaml')

    device = set_device()

    model = LSTMattn(configs['context_size'],
                     hidden_dim=configs['LSTM_config']['hidden_dim'],
                     num_layers=configs['LSTM_config']['num_layers'],
                     output_dim=configs['LSTM_config']['output_dim']
                     ).to(device)

    load_dict = torch.load("models_h5/model_batch_512.h5")

    model.load_state_dict(load_dict['model_state_dict'])

    data_normal_test = read_h5('dataset_free/X_test_normal.h5')
    data_failure_test = read_h5('dataset_free/X_test_failure.h5')

    dataset_test = DatasetWileC(data_normal_test, data_failure_test, context=configs['context_size'])

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=configs['batch_size_test'], shuffle=False)

    model.eval()

    cm, acc = evaluate_matrix_confusion(model.to('cpu'), test_loader, configs['context_size'], 'cpu')  # use estrategies class staticmethods

    print(acc)

    plot_confusion_matrix(cm, ["não defeito", "defeito"], dir_artifacts='artifacts/matrix_confusion.png')
