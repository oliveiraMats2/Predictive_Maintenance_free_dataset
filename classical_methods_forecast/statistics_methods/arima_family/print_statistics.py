import time
import pandas as pd

class PrintStatistics:
    def __call__(self, train, test, func_model, func_avaliable):
        start = time.time()
        y_hat = func_model(train, test)
        end = time.time()
        print(f"tempo de execução: {func_model.__name__} {end - start}")
        data = func_avaliable(test, y_hat, type_model= "multiple")

        mean_abs, smape_loss, mean_square_error = data

        df = pd.DataFrame([mean_abs, smape_loss, mean_square_error], index=["mean_abs",
                                                                            "smape_loss",
                                                                            "mean_square_error"]).T

        print(df.head())
