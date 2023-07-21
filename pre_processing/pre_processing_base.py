if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from data_analisys import DataAnalisysTimeStamp
    from resample_space_time import ResampleSpaceTime

    df = pd.read_csv("../Datasets/dataset_TPV/base_23042023_A/hex/hex.csv")

    # plt.plot(df["InletPressure"].tolist())
    # plt.show()

    resample_up_sampling = ResampleSpaceTime()

    df = resample_up_sampling.resample_up_sampling(df)

    data_analisys_timestamp = DataAnalisysTimeStamp(df)

    # plt.plot(df["InletPressure"].tolist())
    # plt.show()

    # data_analisys_timestamp.plot_frequency_timestamp("hexa", color="blue")

    df_1 = pd.read_csv('../Datasets/dataset_TPV_sensors/wise/x.csv')
    df_2 = pd.read_csv('../Datasets/dataset_TPV_sensors/wise/y.csv')
    df_3 = pd.read_csv('../Datasets/dataset_TPV_sensors/wise/z.csv')

    # Combina as colunas relevantes dos DataFrames
    df = pd.DataFrame()
    df["OAVelocity_x"] = df_1["OAVelocity"].astype('float64')
    df["OAVelocity_y"] = df_2["OAVelocity"].astype('float64')
    df["OAVelocity_z"] = df_3["OAVelocity"].astype('float64')
    df["Time"] = pd.to_datetime(df_1["Time"])

    resample_up_sampling = ResampleSpaceTime()

    df = resample_up_sampling.resample_up_sampling(df)

    data_analisys_timestamp = DataAnalisysTimeStamp(df)

    # data_analisys_timestamp.plot_frequency_timestamp("wise", color="red")

    x = df["OAVelocity_x"].tolist()
    plt.stem(x[:50])
    plt.show()

    df = pd.read_csv("../Datasets/dataset_TPV/base_23042023_A/ite/ite.csv")

    resample_up_sampling = ResampleSpaceTime()

    df = resample_up_sampling.resample_up_sampling(df)

    data_analisys_timestamp = DataAnalisysTimeStamp(df)

    # data_analisys_timestamp.plot_frequency_timestamp("ite", color="green")

