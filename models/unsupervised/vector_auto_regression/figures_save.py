import matplotlib.pyplot as plt


class SaveFigures:
    def __init__(self):
        pass
    @staticmethod
    def save(df_test, df_pred, lim_end=600) -> None:
        features = list(df_test.keys())
        for idx, feature in enumerate(features):
            plt.figure()
            plt.plot(df_test[feature].tolist())
            plt.plot(df_pred[f'{feature}_forecast'].tolist())
            plt.title(f"{feature}")
            plt.xlim(0, lim_end)
            # plt.show()

            plt.savefig(f'{feature}.png')
