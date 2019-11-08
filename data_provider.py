import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ERP_dataset(Dataset):

    def __init__(self, phase):
        super(ERP_dataset, self).__init__()

        self.phase = phase

        training_set_path = "../data/train_data_H5.h5"
        test_set_path = "../data/test_data_H5.h5"

        if self.phase == 'train':
            dataset = h5py.File(training_set_path, 'r')
            self.data = torch.from_numpy(np.array(dataset['data'][:200, :, :]))  # size: [200, 101, 20], type: float64
            self.labels = torch.from_numpy(np.array(dataset['labels'][:200])).squeeze()  # size: [200]
        elif self.phase == 'validate':
            dataset = h5py.File(training_set_path, 'r')
            self.data = torch.from_numpy(np.array(dataset['data'][200:, :, :]))  # size: [38, 101, 20], type: float64
            self.labels = torch.from_numpy(np.array(dataset['labels'][200:])).squeeze()  # size: [38]
        else:
            dataset = h5py.File(test_set_path, 'r')
            self.data = torch.from_numpy(np.array(dataset['data']))  # size: [124, 101, 20], type: float64
            self.labels = torch.from_numpy(np.array(dataset['labels'])).squeeze()  # size: [124]

    def __getitem__(self, index):
        label = 1 if self.labels[index] == 1 else 0
        float32_data = self.data.float()
        return float32_data[index, :, :].unsqueeze(0), label

    def __len__(self):
        return self.labels.size(0)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dataset = ERP_dataset('train')
    print(dataset[0][0].size())

    x = np.arange(0, 101)
    plt.plot(x, dataset[0][0][0, :, 0], color='black')
    print(dataset[0][1])
    plt.plot(x, dataset[1][0][0, :, 0])
    print(dataset[1][1])
    plt.plot(x, dataset[2][0][0, :, 0])
    print(dataset[2][1])
    plt.plot(x, dataset[3][0][0, :, 0], color='black')
    print(dataset[3][1])
    plt.plot(x, dataset[4][0][0, :, 0])
    print(dataset[4][1])
    plt.plot(x, dataset[5][0][0, :, 0])
    print(dataset[5][1])
    plt.plot(x, dataset[6][0][0, :, 0], color='black')
    print(dataset[6][1])
    plt.show()
