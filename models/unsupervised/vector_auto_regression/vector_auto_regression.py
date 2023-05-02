from metrics import avaliable_vector_auto_regressive_model
from trainer_auto_regression import vector_autoRegressive_model_sintetic, vector_autoRegressive_model_real

if __name__ == '__main__':
    data_hex = "/mnt/arquivos_linux/tmp_gustavo_ml/base_17032023_A/ite"

    # ground_truth, pred = vector_autoRegressive_model_sintetic()

    ground_truth, pred = vector_autoRegressive_model_real(f"{data_hex}/payloadITE.csv")

    # avaliable_vector_auto_regressive_model(ground_truth, pred)
