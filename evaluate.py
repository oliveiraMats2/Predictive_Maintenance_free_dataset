import argparse

from tqdm import trange

from tools_wandb import ToolsWandb
from utils.utils import read_yaml
from utils.utils import set_device
from tqdm import tqdm
from constants import *

DEVICE = set_device()


def get_dataset(dataset_configs):
    dataset = FACTORY_DICT["dataset"][list(dataset_configs)[0]](
        **dataset_configs[list(dataset_configs.keys())[0]]
    )

    return dataset


def eval_epoch(model, criterion, loader, epoch):
    model.to(DEVICE)
    model.eval()

    epoch_loss = 0

    with trange(len(loader), desc='Test Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            inputs, labels = sample_batch[0], sample_batch[1]

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            pred_labels = model(inputs)

            loss = criterion(pred_labels, labels)
            epoch_loss += loss.item()

            progress_bar.set_postfix(
                desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(loader):d}, loss: {loss.item():.5f}'
            )

        print(f"\n --- Mean loss {epoch_loss / len(loader)}")


def experiment_factory(configs):
    test_dataset_configs = configs["test_dataset"]
    model_configs = configs["model"]
    criterion_configs = configs["loss"]

    # Construct the dataloaders with any given transformations (if any)
    test_dataset = get_dataset(test_dataset_configs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=configs["test_batch_size"], shuffle=False
    )

    # Build model
    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]]

    return model, test_loader, criterion


def generate_n_samples(model,
                       loader,
                       name_model,
                       iter_n_samples=5000,
                       name_txt='predicted_view_plot.pt') -> None:
    x_test, y_test = loader.dataset[0]

    x_test = x_test.unsqueeze(0)

    save_dict_tensors = {'begin': x_test}

    load_dict = torch.load(name_model)

    model.load_state_dict(load_dict['model_state_dict'])

    model.to(DEVICE)

    with trange(len(loader), desc='evaluate Loop') as progress_bar:
        for idx, sample_batch in zip(progress_bar, loader):

            # if idx == iter_n_samples:
            #     break

            x_test = x_test.to(DEVICE)

            with torch.no_grad():
                predicted = model(x_test).detach().to('cpu')

                predicted = predicted.unsqueeze(1)

                #comentar
                x_test = x_test[:, 1:, :]# transformer old
                # x_test = x_test[:, :, 1:] #transformer new

                # predicted_concat = torch.cat([predicted,
                #                        predicted,
                #                        predicted,
                #                        predicted],axis=2)

                x_test = torch.concat((x_test.to("cpu"), predicted[:, :, -1, :]), dim=1)  # transformer old
                #x_test = torch.concat((x_test.to("cpu"), predicted[:, :, -1, :]), dim=2)  # transformer new

                torch.cuda.empty_cache()

                save_dict_tensors[idx] = predicted[:, :, -1, :]

                # progress_bar.set_postfix(
                #     desc=f'iteration: {idx:d} value: {x_test[:, 395:, 0][0]}'
                # )
                progress_bar.set_postfix(
                    desc=f'iteration: {idx:d} value: {x_test[:, 395:, 0][0]}'
                )
                # progress_bar.set_postfix(
                #     desc=f'iteration: {idx:d} value: {x_test[:, 0, 95:][0]}'
                # )

    torch.save(save_dict_tensors, name_txt)


def generate_n_samples_parallel(model,
                                loader,
                                name_model,
                                name_txt='predicted_view_plot.pt') -> None:
    save_dict_tensors = {}

    load_dict = torch.load(name_model)

    model.load_state_dict(load_dict['model_state_dict'])

    model.to(DEVICE)

    print(f"len dataset test {len(loader.dataset)}")

    parameters = sum(p.numel() for p in model.parameters())

    print(f"quantidade de parametros: {parameters}")

    for i, (x_test, _) in enumerate(tqdm(loader)):
        with torch.no_grad():
            x_test = x_test.to(DEVICE)

            predicted = model(x_test)

            save_dict_tensors[i] = predicted

    torch.save(save_dict_tensors, name_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unsupervised main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    model, test_loader, criterion = experiment_factory(configs)

    name_model = f"{configs['path_to_save_model']}{configs['network']}_{configs['reload_model']['data']}.pt"

    print(name_model)

    f_configurations = {}
    f_configurations = ToolsWandb.config_flatten(configs, f_configurations)

    # run = None
    #
    # if configs['wandb']:
    #     run = wandb.init(project="wile_c_machine_learning_team",
    #                      reinit=True,
    #                      config=f_configurations,
    #                      notes="Testing wandb implementation",
    #                      entity="oliveira_mats",
    #                      dir=None)

    generate_n_samples(model, test_loader, name_model,
                       name_txt=f"models_h5/generate_evaluate/sintetic_generate_data_transformer_9100.pt")

    # generate_n_samples_parallel(model, test_loader, name_model,
    #                            name_txt="sintetic_generate_data_LSTM.pt")
    #
    # f_configurations = {}
    # f_configurations = ToolsWandb.config_flatten(configs, f_configurations)
    #
    # run = None
    #
    # if configs['wandb']:
    #     run = wandb.init(project="wile_c_machine_learning_team",
    #                      reinit=True,
    #                      config=f_configurations,
    #                      notes="Testing wandb implementation",
    #                      entity="oliveira_mats")
    #
    # test_dataset = DatasetSinteticUnsupervisedLSTM(
    #     dir_data=configs["DatasetSinteticUnsupervisedLSTM"]["dir_data"],
    #     context=configs["DatasetSinteticUnsupervisedLSTM"]["context"],
    #     stride=configs["DatasetSinteticUnsupervisedLSTM"]["stride"])
    #
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1, shuffle=False
    # )
    #
    # data_predict = torch.load(f"../sintetic_generate_data_LSTM.pt")
    #
    # vet_predict = union_vector_predicted_dict(data_predict)
