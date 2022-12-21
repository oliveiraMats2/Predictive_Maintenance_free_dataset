#Symmetric mean absolute percentage error
def smape_loss(y_pred, target):
    """
    O erro percentual absoluto médio simétrico é uma medida de precisão baseada em erros percentuais.
     Geralmente é definido da seguinte forma:
    :param y_pred:
    :param target:
    :return:
    """
    y_pred = y_pred.squeeze(2)
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()

