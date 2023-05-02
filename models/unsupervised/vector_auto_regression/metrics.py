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


def avaliable_vector_auto_regressive_model(truth, prediction):
    mae = mean_absolute_error(truth, prediction)
    smape = smape_loss(prediction, truth)
    mse = mean_square_error(truth, prediction)

    print(f"Mean absolut error: {mae}")
    print(f"smape_loss: {smape}")
    print(f"mean absolute error: {mse}")