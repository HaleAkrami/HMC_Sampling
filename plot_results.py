"""demonstrating some utilties in the starter code"""
import argparse
import os

import jax
import matplotlib.image
import matplotlib.pyplot
import numpy

import NPEET.npeet.entropy_estimators
from os import walk
from utils.metrics import get_discretized_tv_for_image_density
from utils.density import continuous_energy_from_image, prepare_image, sample_from_image_density, \
    continuous_energy_from_image_numpy
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
import random
from scipy.interpolate import interp1d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="results/final_res")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--sampler", default="HMC")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--mh_reject", type=bool, default=True)
    args = parser.parse_args()

    seed=args.seed
    if t.cuda.is_available():
        device = t.device('cuda')
        t.cuda.manual_seed_all(seed)
    else:
        device = t.device('cpu')
        print('Warning, no CUDA detected.')

    args.result_folder = args.result_folder + '_' + str(args.epsilon) + '_' + str(args.k) + '_' + args.sampler
    key = jax.random.PRNGKey(0)
    os.makedirs(f"{args.result_folder}", exist_ok=True)

    # load some image
    img = matplotlib.image.imread('./data/labrador.jpg')

    # convert to energy function
    # first we get discrete energy and density values
    density, energy = prepare_image(
        img, crop=(10, 710, 240, 940), white_cutoff=225, gauss_sigma=3, background=0.01
    )

    # create energy fn and its grad
    x_max, y_max = density.shape
    xp = jax.numpy.arange(x_max)
    yp = jax.numpy.arange(y_max)
    zp = jax.numpy.array(density)

    # generate samples from true distribution

    burn_samples=100
    seed_list = []
    samples= None
    for (dirpath, dirnames, filenames) in walk(args.result_folder):
        seed_list.extend(filenames)
        break
    burn_samples = 1
    for  i in seed_list:

        current=np.load(args.result_folder+'/'+i)
        if samples is None:
            samples= np.copy( current[burn_samples:, :])
        else:
            samples = numpy.concatenate((samples, current[burn_samples:, :]), axis=0)


    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(samples[:, 1], samples[:, 0], s=0.5, alpha=0.5)
    ax.set_xlim([0, 700])
    ax.set_ylim([0, 700])
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/fig/labrador_sampled.png")

    # generate another set of samples from true distribution, to demonstrate comparisons
    key, subkey = jax.random.split(key)
    second_samples = sample_from_image_density(len(samples), density, subkey)
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(numpy.array(second_samples[:, 1]), numpy.array(second_samples[:, 0]), s=0.5, alpha=0.5)
    ax.set_xlim([0, 700])
    ax.set_ylim([0, 700])
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/fig/true_sampled.png")

    # We have samples from two distributions. We use NPEET package to compute kldiv directly from samples.
    # NPEET needs nxd tensors
    t=numpy.array(second_samples)
    kldiv = NPEET.npeet.entropy_estimators.kldiv(samples, t)
    print(f"KL divergence is {kldiv}")

    # TV distance between discretized density
    # The discrete density bin from the image give us a natural scale for discretization.
    # We compute discrete density from sample at this scale and compute the TV distance between the two densities
    tv_dist = get_discretized_tv_for_image_density(
        numpy.asarray(density), numpy.asarray(samples), bin_size=[7, 7]
    )
    print(f"TV distance is {tv_dist}")


