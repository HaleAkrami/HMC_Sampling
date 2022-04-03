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
import os
import torch as t
from esh import datasets, utils, samplers
import random



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="results")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--sampler", default="HMC")
    parser.add_argument("--epsilon",type=float,  default=0.1) # 0.01 0.1 0.2
    parser.add_argument("--k", type=int, default=50) #5 50 100
    parser.add_argument("--step_size", type=int, default=50*100000) # 100000
    parser.add_argument("--mh_reject", type=bool, default=True)

    args = parser.parse_args()
    seed =args.seed   # adjusted to get non-overlapping lines
    t.manual_seed(seed)
    if t.cuda.is_available():
        device = t.device('cuda')
        t.cuda.manual_seed_all(seed)
    else:
        device = t.device('cpu')
        print('Warning, no CUDA detected.')

    args.result_folder=args.result_folder+'_'+str(args.epsilon)+'_'+str(args.k)+'_'+args.sampler
    key = jax.random.PRNGKey(0)
    os.makedirs(f"{args.result_folder}", exist_ok=True)
    os.makedirs(f"{args.result_folder}/fig", exist_ok=True)

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
    fig.savefig(f"{args.result_folder}/fig/labrador_density.png")

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(energy)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/fig/labrador_energy.png")

    # create energy fn and its grad
    x_max, y_max = density.shape

    xp_np = t.from_numpy(numpy.arange(x_max))
    yp_np = t.from_numpy(numpy.arange(y_max))
    zp_np = t.from_numpy(numpy.array(density))

    # You may use fill value to enforce some boundary conditions or some other way to enforce boundary conditions
    energy_fn = lambda coord: continuous_energy_from_image_numpy(coord, xp_np, yp_np, zp_np, fill_value=0)


    # generate samples from true distribution
    if args.sampler== "HMC":
        sampler_list = [("HMC", samplers.hmc_integrate, {'epsilon': args.epsilon, 'k': args.k, 'mh_reject': args.mh_reject})]
    elif args.sampler== "Ndynamic":
        sampler_list = [("Ndynamic", samplers.newton_dynamics, {'epsilon':args.epsilon})]
    num_samples = args.step_size
    for k, (sampler_name, sampler, kwargs) in enumerate(sampler_list):
        x0 = t.tensor(
            [random.uniform(0,700), random.uniform(0,700)])  # Choose different modes for initialization, so no overlap
        energy = energy_fn

        xs, vs, ts = sampler(energy, x0, num_samples , **kwargs)
        save_des=args.result_folder+'/xs_'+str(args.seed)
    numpy.save(save_des, numpy.array(xs))




