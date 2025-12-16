# Copyright 2025 Rohde & Schwarz

import matplotlib.pyplot as plt
import numpy as np

def display_intermediate_samples(samples, grid, iteration, dim):
    """
    displays a figures that plots 9 samples

    Parameters:
    - samples: samples with dimension 9 x sdim/hdim (cplx numpy)
    - dim: dimension of the samples (typically sdim)
    - grid: list(!) of arrays (in numpy) describing the gridpoints used for the dictionary
    - iteration: iteration counter of the training
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    fig.suptitle(f'Squared absolute value of 9 randomly sampled channels after {iteration} iterations', fontsize=16,fontweight='bold')

    if len(grid) == 1:
        samples = np.abs(samples.reshape((9, dim))) ** 2
        x_ticks = np.array([0, (dim - 1) // 2, dim - 1])
        y_ticks = np.array([])
        for n, ax in enumerate(axes):
            im = ax.plot(samples[n, :, :], cmap='viridis', aspect='auto')
            ax.set_title(f"sample {n + 1}")
            # Set tick positions and labels
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([str(-round(grid[0][0],2)), '0', str(round(grid[0][-1],2))])
            ax.set_xlabel("angle")
        plt.show()

    elif len(grid) == 2:
        sdim_single = int(np.sqrt(dim))
        samples = np.abs(samples.reshape((9, sdim_single, sdim_single))) ** 2
        x_ticks = np.array([0, (sdim_single - 1) // 2, sdim_single - 1])
        y_ticks = np.array([0, sdim_single - 1])
        axes = axes.flatten()
        for n, ax in enumerate(axes):
            im = ax.imshow(samples[n, :, :], cmap='viridis', aspect='auto')
            ax.set_title(f"sample {n + 1}")
            # Set tick positions and labels
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([str(-round(grid[1][0],2)), '0', str(round(grid[1][-1] - grid[1][-1] / sdim_single, 2))])
            ax.set_yticklabels(['0', str(np.format_float_scientific(grid[0][-1] - grid[0][-1] / sdim_single,2))])
            ax.set_xlabel("doppler in Hz")
            ax.set_ylabel("delay in seconds")
            fig.colorbar(im, ax=axes[n])  # Add colorbar to each plot
        # Adjust layout
        plt.tight_layout()
        plt.show()


def save_intermediate_samples(samples, grid, iteration, dim, dir_path):
    """
    saves samples as standalone pngs and numpy files in the given directory

    Parameters:
    - samples: samples with dimension 9 x sdim/hdim
    - dim: dimension of the samples (typically sdim)
    - grid: list(!) of arrays describing the gridpoints used for the dictionary
    - iteration: iteration counter of the training
    - dir_path: string for the directory, in which the samples are stored
    """
    n_samples = samples.shape[0]
    samples = np.abs(samples)**2
    # save numpy array
    np.save(dir_path + 'samples_iter' + str(iteration), samples)
    # save images
    if len(grid) == 1:
        for n in range(n_samples):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(f'sample:{n}, iter: {iteration}', fontsize=20)
            ax.plot(grid[0], samples[n, :], linewidth=2.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Angle", fontsize=20)
            plt.savefig(dir_path + '/im' + str(n) + '_iter' + str(iteration) + '.png')
            plt.tight_layout()
            plt.close()

    elif len(grid) == 2:
        sdim_single = int(np.sqrt(dim))
        samples = samples.reshape((n_samples, sdim_single, sdim_single))
        for n in range(n_samples):
            plt.figure(figsize=(15, 5))
            plt.title(f'sample:{n}, iter: {iteration}')
            plt.imshow(samples[n, :, :])
            plt.savefig(dir_path + '/im' + str(n) + '_iter' + str(iteration) + '.png')
            plt.close()