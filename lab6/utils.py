
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

def plot_choices(q, epsilon, choice_fn, n_steps=1000, rng_seed=1):
    np.random.seed(rng_seed)
    counts = np.zeros_like(q, dtype=float)
    for t in range(n_steps):
        action = choice_fn(q, epsilon)
        counts[action] += 1

    # Normalize counts to get percentages
    counts /= n_steps

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(range(len(q)), counts)
    ax.set_ylabel('% chosen', fontsize=12, fontweight='bold')
    ax.set_xlabel('Action', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(q)))
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)
    
    # Set bold frame
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.show()

def plot_multi_armed_bandit_results(results):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))

    # Total Reward plot
    ax1.plot(results['rewards'])
    ax1.set(title=f"Total Reward: {np.sum(results['rewards']):.2f}",
            xlabel='Step', ylabel='Reward')
    ax1.title.set_weight('bold')
    ax1.xaxis.label.set_weight('bold')
    ax1.yaxis.label.set_weight('bold')
    ax1.tick_params(axis='both', which='both', labelsize=10, width=2)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    
    # Action Values plot
    ax2.plot(results['qs'])
    ax2.set(xlabel='Step', ylabel='Value')
    ax2.xaxis.label.set_weight('bold')
    ax2.yaxis.label.set_weight('bold')
    ax2.tick_params(axis='both', which='both', labelsize=10, width=2)
    ax2.legend(range(len(results['mu'])))
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    # Latent and Learned Values plot
    ax3.plot(results['mu'], label='Latent')
    ax3.plot(results['qs'][-1], label='Learned')
    ax3.set(xlabel='Action', ylabel='Value')
    ax3.xaxis.label.set_weight('bold')
    ax3.yaxis.label.set_weight('bold')
    ax3.tick_params(axis='both', which='both', labelsize=10, width=2)
    ax3.legend()
    for spine in ax3.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.show()

def plot_parameter_performance(labels, fixed, trial_rewards, trial_optimal):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

    # Average Reward plot
    ax1.plot(np.mean(trial_rewards, axis=1).T)
    ax1.set(title=f'Average Reward ({fixed})', xlabel='Step', ylabel='Reward')
    ax1.title.set_weight('bold')
    ax1.xaxis.label.set_weight('bold')
    ax1.yaxis.label.set_weight('bold')
    ax1.tick_params(axis='both', which='both', labelsize=10, width=2)
    ax1.legend(labels)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    # Performance plot
    ax2.plot(np.mean(trial_optimal, axis=1).T)
    ax2.set(title=f'Performance ({fixed})', xlabel='Step', ylabel='% Optimal')
    ax2.title.set_weight('bold')
    ax2.xaxis.label.set_weight('bold')
    ax2.yaxis.label.set_weight('bold')
    ax2.tick_params(axis='both', which='both', labelsize=10, width=2)
    ax2.legend(labels)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.show()