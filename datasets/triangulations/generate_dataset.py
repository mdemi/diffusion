from cytools import Polytope
from cytools.triangulation import *
import ray # We use ray for parallelization. Install with pip install ray.
import numpy as np
from tqdm import tqdm
import os
import h5py as h5

# Paramaters
image_size = 8
sample_size = 2**20
threads = 32
local_storage_dir = "/home/md775/LocalStorage/MLProjects/Diffusion/" # Change this to your storage directory
dataset_dir = local_storage_dir + "Datasets/RegularTriangulations/"
vertices = np.array([[0, 0], [0, image_size-1], [image_size-1, image_size-1], [image_size-1, 0]])

# Functions
def create_heights_image(points, heights):
    image = np.zeros((image_size, image_size))
    for pt, h in zip(points, heights):
        image[pt[0], pt[1]] = h
    return image

@ray.remote
def sample_triangulations(N, num_flips=1):
    sample = []
    pbar = tqdm(total=N)
    temp_tri = poly.triangulate(make_star=False)
    tri_ctr = 0
    while tri_ctr<N:
        temp_tri = temp_tri.random_flips(num_flips, only_fine=True, only_regular=True, only_star=False)
        heights = temp_tri.secondary_cone().tip_of_stretched_cone(1)
        heights = heights / np.linalg.norm(heights)
        sample.append(create_heights_image(temp_tri.points(), heights))
        tri_ctr += 1
        pbar.update(1)
    return sample

if __name__ == "__main__":
    ray.init(num_cpus=threads)
    poly = Polytope(vertices)
    futures = [sample_triangulations.remote(sample_size//threads+1) for _ in range(threads)]
    sample = np.array(ray.get(futures))
    sample = np.concatenate(sample, axis=0)
    sample = sample[:sample_size]
    ray.shutdown()

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    # Save as HDF5
    with h5.File(dataset_dir + "dataset.h5", "w") as f:
        f.create_dataset("height_images", data=sample)
        f.create_dataset("polytope", data=poly.points())