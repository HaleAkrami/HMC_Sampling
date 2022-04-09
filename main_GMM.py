"""Test a variety of samplers against a variety of datasets. Used to generate MMD plots and ESS table for the paper."""
import os
import numpy as np
import pickle
import torch as t
import json
from torch.utils.tensorboard import SummaryWriter
from esh import datasets, utils, samplers
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.pyplot
import NPEET.npeet.entropy_estimators

plt.style.use('seaborn-paper')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


if __name__ == '__main__':
    # Seed for reproducibility
    seed = 4  # adjusted to get non-overlapping lines
    t.manual_seed(seed)
    if t.cuda.is_available():
        device = t.device('cuda')
        t.cuda.manual_seed_all(seed)
    else:
        device = t.device('cpu')
        print('Warning, no CUDA detected.')

    # Logging
    dataset = 'GMM_fig'
    
    log_root = os.path.join("/content/drive/MyDrive/", 'phd_docs/HW_2/{}/'.format(dataset))
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    last_run = max([0] + [int(k) for k in os.listdir(log_root) if k.isdigit()])
    log_dir = os.path.join(log_root, '{0:04d}'.format(last_run + 1))  # Each run log gets a new directory
    os.makedirs(log_dir)
    for folder in ['figs', 'code', 'checkpoints']:
        os.mkdir(os.path.join(log_dir, folder))
    os.system('cp *.py {}/code/'.format(log_dir))
    print("Visualize logs using: tensorboard --logdir={0}".format(log_dir))
    writer = SummaryWriter(log_dir)

    sampler_list = [
                      # time scaled
                      ('NUTS', samplers.nuts, {})
                      ('MALA 0.1', samplers.hmc_integrate, {'epsilon': 0.01, 'k': 5, 'mh_reject': True}),
                      ('MALA 0.1', samplers.hmc_integrate, {'epsilon': 0.15, 'k': 5, 'mh_reject': True}),
                      ('MALA 0.1', samplers.hmc_integrate, {'epsilon': 0.1, 'k': 5, 'mh_reject': True})
                   ]

    n_steps = 10000  # number of gradient steps

    save = []
    e_name, e_model = ('2D MOG-prior', datasets.ToyDataset(toy_type='gmm') )
    result = {}
    
    
    for k, (sampler_name, sampler, kwargs) in enumerate(sampler_list):
        
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        fig.set_size_inches(8, 8, forward=True)
        ax.set_aspect('equal')
        plt.axis('off')
        fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)
        x0 = 0.5 * t.tensor([np.sin(k * np.pi / 4), np.cos(k * np.pi / 4)])  # Choose different modes for initialization, so no overlap
        
        
        energy = e_model.energy

        xs, vs, ts, reject_count = sampler(energy, x0, n_steps, **kwargs)
        #xs, vs, ts = sampler(energy, x0, n_steps, **kwargs)
        print('reject_count', reject_count)
        exp_string = '{}_{}_{}'.format(e_name, sampler_name,kwargs)
        
        print(exp_string)
        colors = ['tab:blue', 'tab:green', 'cyan', 'magenta', 'blue', 'darkgreen', 'cyan', 'lime']
        ax.scatter(xs[:, 0], xs[:, 1], c=colors[k], lw=1.5)
      
        r = 1
        n_x = 100
        alpha = 0.7
        xv, yv = np.meshgrid(np.linspace(-r, r, n_x), np.linspace(-r, r, n_x))
        x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
        with t.no_grad():
            energy_all = energy(t.tensor(x_grid, dtype=xs.dtype)).cpu().numpy()
    
        #e_model.plot_toy_density(plot_truth=True, x_s_t=xs,save_path="HMC_{}_{}.pdf".format(kwargs["k"],kwargs["epsilon"]))
        e_model.plot_toy_density(plot_truth=True, x_s_t=xs,save_path="NUTS.pdf")
        second_samples = e_model.sample_data(n_steps)
        print(len(xs))
        print(len(second_samples))
        kldiv = NPEET.npeet.entropy_estimators.kldiv(second_samples,xs[1:,:])
        print(f"KL divergence is {kldiv}")
        
        e_history_grid = energy_all.reshape((n_x, n_x))
        xs_grid = x_grid[:, 0].reshape((n_x, n_x))
        ys_grid = x_grid[:, 1].reshape((n_x, n_x))
        p_grid = np.exp(-e_history_grid) / np.sum(np.exp(-e_history_grid))
        grid = [-4. + 0.1 + i + np.log(p_grid.max()) for i in range(5)]
        filename = '{}/figs/samples_{}.png'.format(log_dir, exp_string)
        fig.savefig(filename, transparent=True)
        ax.contourf(xs_grid, ys_grid, np.log(p_grid), grid,cmap="OrRd", zorder=0, alpha=alpha)
        filename = '{}/figs/trajectory_{}.png'.format(log_dir, exp_string)
        fig.savefig(filename, transparent=True)
        result[exp_string] = {"KL":kldiv,"reject_count":reject_count}
        #result[exp_string] = {"KL":kldiv,"reject_count":reject_count}
        plt.close(fig)
    with open('metrics_HMC_2.json', 'w') as outfile:
      json.dump(result,outfile)
