"""demonstrating some utilties in the starter code"""
import argparse
import os

import jax
import matplotlib.image
import matplotlib.pyplot
import numpy

import NPEET.npeet.entropy_estimators
from utils.metrics import get_discretized_tv_for_image_density
from utils.density import continuous_energy_from_image, prepare_image, sample_from_image_density,continuous_energy_from_image_numpy
from esh.datasets import ToyDataset
from esh.samplers import hmc_integrate
import os
import numpy as np
import pickle
import torch as t
from torch.utils.tensorboard import SummaryWriter
from esh import datasets, utils, samplers
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.interpolate import interp1d



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="results")
    seed = 4  # adjusted to get non-overlapping lines
    t.manual_seed(seed)
    if t.cuda.is_available():
        device = t.device('cuda')
        t.cuda.manual_seed_all(seed)
    else:
        device = t.device('cpu')
        print('Warning, no CUDA detected.')

    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    os.makedirs(f"{args.result_folder}", exist_ok=True)

    # load some image
    img = matplotlib.image.imread('./data/labrador.jpg')

    # plot and visualize
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    matplotlib.pyplot.show()

    # convert to energy function
    # first we get discrete energy and density values
    density, energy = prepare_image(
        img, crop=(10, 710, 240, 940), white_cutoff=225, gauss_sigma=3, background=0.01
    )

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(density)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/labrador_density.png")

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(energy)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/labrador_energy.png")

    # create energy fn and its grad
    x_max, y_max = density.shape
    xp = jax.numpy.arange(x_max)
    yp = jax.numpy.arange(y_max)
    zp = jax.numpy.array(density)

    xp_np = t.from_numpy(numpy.arange(x_max))
    yp_np = t.from_numpy(numpy.arange(y_max))
    zp_np = t.from_numpy(numpy.array(density))

    # You may use fill value to enforce some boundary conditions or some other way to enforce boundary conditions
    energy_fn = lambda coord: continuous_energy_from_image_numpy(coord, xp_np, yp_np, zp_np, fill_value=0)
    #energy_fn_grad = jax.grad(energy_fn)

    # NOTE: JAX makes it easy to compute fn and its grad, but you can use any other framework.


    # generate samples from true distribution
    sampler_list = [('MALA 0.1', samplers.hmc_integrate, {'epsilon': 0.1, 'k': 5, 'mh_reject': True})]
    num_samples = 50*100000
    for k, (sampler_name, sampler, kwargs) in enumerate(sampler_list):
        x0 = 0.5 * t.tensor(
            [np.sin(k * np.pi / 4), np.cos(k * np.pi / 4)])  # Choose different modes for initialization, so no overlap
        energy = energy_fn

        xs, vs, ts = sampler(energy, x0+300, num_samples , **kwargs)
    # (scatter) plot the samples with image in background
        #FIXME remove burnin samples
        burn_samples=100
        samples=xs[burn_samples:,:]
        fig = matplotlib.pyplot.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(numpy.array(samples)[:, 1], numpy.array(samples)[:, 0], s=0.5, alpha=0.5)
        #ax.imshow(density, alpha=0.3)
        matplotlib.pyplot.show()
        fig.savefig(f"{args.result_folder}/labrador_sampled.png")

    # generate another set of samples from true distribution, to demonstrate comparisons
    key, subkey = jax.random.split(key)
    second_samples = sample_from_image_density(num_samples, density, subkey)

    # We have samples from two distributions. We use NPEET package to compute kldiv directly from samples.
    # NPEET needs nxd tensors
    kldiv = NPEET.npeet.entropy_estimators.kldiv(samples, second_samples)
    print(f"KL divergence is {kldiv}")

    # TV distance between discretized density
    # The discrete density bin from the image give us a natural scale for discretization.
    # We compute discrete density from sample at this scale and compute the TV distance between the two densities
    tv_dist = get_discretized_tv_for_image_density(
        numpy.asarray(density), numpy.asarray(samples), bin_size=[7, 7]
    )
    print(f"TV distance is {tv_dist}")


