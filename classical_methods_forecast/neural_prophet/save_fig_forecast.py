import pandas as pd
import matplotlib.pyplot as plt


class SaveFigForecast:
    def __call__(self, ds: pd.core.series.Series,
                 y_truth: list,
                 y_hat: list,
                 **configs: dict) -> None:

        x_axis: str = configs["x_axis"]
        y_axis: str = configs["y_axis"]
        title: str = configs["title"]

        plt.scatter(ds, y_hat, s=configs["length_circle"], color="cornflowerblue", label="Regression")
        plt.scatter(ds, y_truth, s=configs["length_circle"], color="black", label="Real")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(title)
        plt.grid(True)
        plt.show()
        plt.savefig(f'preview_save_neural_prophet_/forecast_{configs["select_feature"]}.png')
