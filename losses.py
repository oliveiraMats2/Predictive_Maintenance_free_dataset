from soft_dtw_cuda import SoftDTW

sdtw = SoftDTW(use_cuda=True, gamma=0.1)


# #https://towardsdatascience.com/time-series-forecast-error-metrics-you-should-know-cc88b8c67f27centage error
def smape_loss(y_pred, target):
    """
    O erro percentual absoluto médio simétrico é uma medida de precisão baseada em erros percentuais.
     Geralmente é definido da seguinte forma:
    :param y_pred:
    :param target:
    :return:
    """
    # y_pred = y_pred.squeeze(2)
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()


def soft_dtw(y_pred, target):
    loss = sdtw(y_pred.unsqueeze(2), target.unsqueeze(2))
    return loss.mean()
