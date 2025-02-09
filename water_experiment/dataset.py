import os
import numpy as np
import torch

from deprecated.eqnode.particle_utils import remove_mean
from deprecated.eqnode.test_systems import MultiDoubleWellPotential
from deprecated.eqnode.train_utils import IndexBatchIterator, BatchIterator


def get_data(args, partition, batch_size):
    if args.data == 'wat2_gaff':
        return get_data_wat(args.n_data, partition, batch_size, 6, "wat2_gaff")
    elif args.data == 'wat2_arrow':
        return get_data_wat(args.n_data, partition, batch_size, 6, "wat2_arrow")
    elif args.data == 'wat5_gaff':
        return get_data_wat(args.n_data, partition, batch_size, 15, "wat5_gaff")
    elif args.data == 'wat5_arrow':
        return get_data_wat(args.n_data, partition, batch_size, 15, "wat5_arrow")
    else:
        raise ValueError


def get_data_wat(n_data, partition, batch_size, n_particles_val=6, data_prefix = "wat2_gaff"):
    
    n_particles = n_particles_val
    n_dimension = 3
    dim = n_particles * n_dimension

    if partition == 'train':
        data = np.load(os.path.join("water_experiment","data",data_prefix + ".npy"))[0:n_data]
    elif partition == 'val':
        data = np.load(os.path.join("water_experiment","data",data_prefix + ".npy"))[n_data:n_data + 1000]
    elif partition == 'test':
        data = np.load(os.path.join("water_experiment","data",data_prefix + ".npy"))[n_data:n_data + 1000]
    elif partition == 'all':
        data = np.load(os.path.join("water_experiment","data",data_prefix + ".npy"))
    else:
        raise Exception("Wrong partition")

    data = data.reshape(-1, dim)
    data = torch.Tensor(data)
    # data = remove_mean(data, n_particles, dim // n_particles)  #IGOR_TMP - don't to remove_mean

    #if partition == 'train':
    #    data = data[idx[:n_data]].clone()
    
    print(f"get_data_wat() {n_data=} {partition=} {len(data)=} {batch_size=} {data.shape=}")

    batch_iter = BatchIterator(len(data), batch_size)

    return data, batch_iter


def plot_data(sample):
    import matplotlib.pyplot as plt
    x = sample[:, 0]
    y = sample[:, 1]
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    n_samples = 100
    data, _ = get_data_dw4(n_samples, 'train')
    data = data.view(n_samples, 4, 2)
    for i in range(n_samples):
        plot_data(data[i])
