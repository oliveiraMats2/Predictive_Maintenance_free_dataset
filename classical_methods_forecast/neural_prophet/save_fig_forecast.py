import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        plt.legend()
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(title)
        plt.grid(True)
        plt.savefig(f'preview_save_neural_prophet_/forecast_{configs["select_feature"]}_just_blue.png', dpi=800)
        plt.show()

    def plot_analysis_data(self, df_truth):
        y = df_truth["y"].tolist()
        x = list(range(len(y)))

        plt.scatter(x, y, s=0.5)
        plt.show()

    def plot_presentation(self, ds_train: pd.core.series.Series,
                          ds_test: pd.core.series.Series,
                          y_truth: list,
                          y_hat: list,
                          **configs: dict) -> None:
        plt.rcParams["figure.figsize"] = (20, 6)
        plt.rcParams["figure.autolayout"] = True

        x_axis: str = configs["x_axis"]
        y_axis: str = configs["y_axis"]
        title: str = configs["title"]
        if "y_lim" in configs.keys():
            y_lim: int = configs["y_lim"]
            plt.ylim((y_lim[0], y_lim[1]))

        plt.scatter(ds_test, y_hat, s=configs["length_circle"], color="cornflowerblue", label="Regression")
        plt.scatter(ds_train, y_truth, s=configs["length_circle"], color="black", label="Real")
        plt.legend()
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(title)
        plt.grid(True)
        plt.savefig(f'preview_plots_inference/forecast_{configs["select_feature"]}_just_blue.png', dpi=800)
        plt.show()
