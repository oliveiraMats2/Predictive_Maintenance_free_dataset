import numpy as np

def smape_loss(y_pred, target):
    # y_pred = y_pred.squeeze(2)
    loss = 2 * abs((y_pred - target)) / (abs(y_pred) + abs(target) + 1e-8)
    return loss.mean() * 100


def mean_absolute_error(slice_data_true, slice_data_pred):
    abs_data = abs(slice_data_true - slice_data_pred)
    return abs_data.mean()


# Mean absolut error

def mean_square_error(slice_data_true, slice_data_pred):
    sum_square = 0
    for i, elem in enumerate(slice_data_true):
        sum_square += (slice_data_true[i] - slice_data_pred[i]) ** 2

    sum_square = sum_square / (i + 1)

    return sum_square


class MeasureMultiSensors:
    @staticmethod
    def mean_absolute_error(df_truth, df_prediction):
        dict_result = {}
        features = list(df_truth.keys())
        for idx, feature in enumerate(features):
            truth, pred = df_truth[feature].tolist(), df_prediction[f'{feature}_forecast'].tolist()
            dict_result[feature] = mean_absolute_error(np.array(truth), np.array(pred))

        return dict_result

    @staticmethod
    def smape_loss(df_truth, df_prediction):
        dict_result = {}
        features = list(df_truth.keys())
        for idx, feature in enumerate(features):
            truth, pred = df_truth[feature].tolist(), df_prediction[f'{feature}_forecast'].tolist()
            dict_result[feature] = smape_loss(np.array(truth), np.array(pred))

        return dict_result

    @staticmethod
    def mean_square_error(df_truth, df_prediction):
        dict_result = {}
        features = list(df_truth.keys())
        for idx, feature in enumerate(features):
            truth, pred = df_truth[feature].tolist(), df_prediction[f'{feature}_forecast'].tolist()
            dict_result[feature] = mean_square_error(np.array(truth), np.array(pred))

        return dict_result


def avaliable_vector_auto_regressive_model(truth, prediction, type_model="single"):
    if type_model == "single":
        mae = mean_absolute_error(truth, prediction)
        smape = smape_loss(prediction, truth)
        mse = mean_square_error(truth, prediction)

        print(f"Mean absolut error: {mae}")
        print(f"smape_loss: {smape}")
        print(f"mean mse: {mse}")

        return mae, smape, mse

    if type_model == "multiple":
        dict_mae = MeasureMultiSensors.mean_absolute_error(truth, prediction)
        dict_smape = MeasureMultiSensors.smape_loss(truth, prediction)
        dict_mse = MeasureMultiSensors.mean_square_error(truth, prediction)

        return dict_mae, dict_smape, dict_mse
