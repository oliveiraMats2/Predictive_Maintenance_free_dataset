import matplotlib.pyplot as plt


class SaveFigures:

    @staticmethod
    def highest_value(x, y):
        if x > y:
            return x, y
        else:
            return y, x
    @staticmethod
    def save(df_test, df_pred, lim_end=600, sub_path="payloadITE", **kwargs) -> None:
        features = list(df_test.keys())

        first_limiar = kwargs["first_limiar"]

        for idx, feature in enumerate(features):
            plt.figure()

            test = df_test[feature].tolist()
            pred = df_pred[f'{feature}_forecast'].tolist()

            high, low = SaveFigures.highest_value(test[first_limiar], pred[first_limiar])



            plt.plot(test)
            plt.plot(pred)

            plt.text(first_limiar,
                     (low + (high-low)/2),
                     f'{(high - low):.2f}',
                     ha='center',
                     va='center',
                     rotation='vertical',
                     backgroundcolor='white')

            plt.vlines(x=first_limiar,
                       ymin=low,
                       ymax=high,
                       colors='green',
                       ls=':')

            plt.title(f"{feature}")
            plt.xlim(0, lim_end)
            # plt.show()

            plt.savefig(f'{sub_path}/{feature}.png')