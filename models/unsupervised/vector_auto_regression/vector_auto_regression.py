from metrics import avaliable_vector_auto_regressive_model
from trainer_auto_regression import vector_autoRegressive_model_sintetic, vector_autoRegressive_model_real
import pandas as pd

if __name__ == '__main__':
    data_ite = "/mnt/arquivos_linux/wile_C/Predictive_Maintenance_free_dataset/Datasets/dataset_TPV/base_17032023_A/ite"
    # data_hex = "/mnt/arquivos_linux/wile_C/Predictive_Maintenance_free_dataset/Datasets/dataset_TPV/base_17032023_A/hex"
    # data_wise = "/mnt/arquivos_linux/wile_C/Predictive_Maintenance_free_dataset/Datasets/dataset_TPV/base_17032023_A/wise"

    # ground_truth, pred = vector_autoRegressive_model_sintetic()

    name = "payloadITE.csv"
    # name = "payloadHex.csv"
    # name = "x.csv"
    ground_truth, pred = vector_autoRegressive_model_real(f"{data_ite}/{name}",
    # ground_truth, pred = vector_autoRegressive_model_real(f"{data_hex}/{name}",
                                                          window=50,
                                                          steps=200,
                                                          order=9,
                                                          calcs_range_model=10,
                                                          first_limiar=100,
                                                          sub_path="payloadHex")

    mean_abs, smape_loss, mean_square_error = avaliable_vector_auto_regressive_model(ground_truth,
                                                                                     pred,
                                                                                     "multiple")

    df = pd.DataFrame([mean_abs, smape_loss, mean_square_error], index=["mean_abs",
                                                                        "smape_loss",
                                                                        "mean_square_error"]).T

    df.index.name = "variables"

    df.to_csv(f"result_multi_sensor_{name}")


