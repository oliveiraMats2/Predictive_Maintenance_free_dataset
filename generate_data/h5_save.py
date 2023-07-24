import h5py


class SaveData:
    @staticmethod
    def save_data(data, dir_data='../Datasets/sintetic_data/train_compressor_data.h5'):
        with h5py.File(dir_data, 'w') as h5f:
            h5f.create_dataset('data_train', data=data)
