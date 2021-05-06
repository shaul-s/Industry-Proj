import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns

sns.set(style='whitegrid')


def plot_radial_distortion(k1, k2, k3, ax):
    max = 10
    x, y = np.meshgrid(np.arange(-max, max, .5), np.arange(-max, max, .5))
    r2 = x ** 2 + y ** 2
    dr = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    v, u = x * dr, y * dr

    ###
    x_corr = x + v
    y_corr = y + u
    ###

    # ax.quiver(x, y, - x_corr + x, - y_corr + y, color='r')
    ax.quiver(x, y, v, u, color='r')

    # display
    # plt.show()


def plot_tangential_distortion(p1, p2, ax):
    max = 10
    x, y = np.meshgrid(np.arange(-max, max, .5), np.arange(-max, max, .5))
    r2 = x ** 2 + y ** 2
    xy = 2 * x * y
    ax.quiver(x, y, xy * p1 + (r2 + 2 * x ** 2) * p2, xy * p2 + (r2 + 2 * y ** 2) * p1, color='g')

    # display
    # plt.show()


def plot_diffscale_distortion(b1, b2, ax):
    max = 10
    x, y = np.meshgrid(np.arange(-max, max, .5), np.arange(-max, max, .5))
    # r2 = x ** 2 + y ** 2
    ax.quiver(x, y, b1 * x + b2 * y, 0, color='b')

    # display
    # plt.show()


def plot_all_distortions(k1, k2, k3, p1, p2, b1, b2, ax):
    max = 10
    x, y = np.meshgrid(np.arange(-max, max, .5), np.arange(-max, max, .5))
    r2 = x ** 2 + y ** 2
    dr = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xy = x * y

    v = x * dr + p1 * (r2 + 2 * x ** 2) + 2 * p2 * xy + b1 * x + b2 * y
    u = y * dr + p2 * (r2 + 2 * y ** 2) + 2 * p1 * xy

    ax.quiver(x, y, v, u, color='b')


def plot_dr2r(k1, k2, k3, p1, p2, b1, b2):
    x, y = np.arange(0, 200, 0.1), np.arange(0, 200, 0.1)
    r2 = x ** 2 + y ** 2
    dr = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xy = x * y

    v = x * dr + p1 * (r2 + 2 * x ** 2) + 2 * p2 * xy + b1 * x + b2 * y
    u = y * dr + p2 * (r2 + 2 * y ** 2) + 2 * p1 * xy

    correction = np.sqrt(v ** 2 + u ** 2)

    plt.plot(np.sqrt(r2), correction)
    plt.xlabel("r'")
    plt.ylabel("dr'")

    # fig = px.line(x=r, y=dr)

    # showing the plot
    # fig.show()


def plot_before_after(k1, k2, k3, p1, p2, b1, b2, orig_samples, residuals, ax, scale=1):
    orig_samples = pd.read_csv(orig_samples, index_col=0)
    residuals = pd.read_csv(residuals, index_col=0)
    orig_samples = orig_samples.loc[:0.3 * orig_samples.size]
    residuals = residuals.loc[:0.3 * residuals.size]

    orig_x = orig_samples.values[::2]
    orig_y = orig_samples.values[1::2]

    residuals_x = residuals.values[::2]
    residuals_y = residuals.values[1::2]

    r2 = orig_x ** 2 + orig_y ** 2
    dr = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xy = orig_x * orig_y

    v = orig_x * dr + p1 * (r2 + 2 * orig_x ** 2) + 2 * p2 * xy + b1 * orig_x + b2 * orig_y
    u = orig_y * dr + p2 * (r2 + 2 * orig_y ** 2) + 2 * p1 * xy

    ax[0].set_title('Before Calibration')
    ax[0].axvline(x=0, c='k', linewidth=0.75)
    ax[0].axhline(y=0, c='k', linewidth=0.75)
    ax[0].scatter(orig_x, orig_y, color='r', s=0.75)
    ax[0].quiver(orig_x, orig_y, v, u, color='b', scale=0.05, linewidth=0.05)
    ax[1].set_title('After Calibration')
    ax[1].axvline(x=0, c='k', linewidth=0.75)
    ax[1].axhline(y=0, c='k', linewidth=0.75)
    ax[1].scatter(orig_x, orig_y, color='r', s=0.75)
    ax[1].quiver(orig_x, orig_y, scale * residuals_x, scale * residuals_y, color='b', scale=0.5, linewidth=0.05)


def compute_corr(Q):
    # computing correlation matrix
    corr = np.zeros(Q.shape)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            corr[i, j] = Q[i, j] / np.sqrt(Q[i, i] * Q[j, j])

    return corr


def plot_calib_corr_map(Q, ax, num_images=50):
    Q = pd.read_csv(Q, index_col=0).values
    full_corr_mat = compute_corr(Q)
    calib_corr_map = np.tril(full_corr_mat[6 * num_images:6 * num_images + 10, 6 * num_images:6 * num_images + 10])

    corr_map_dat = pd.DataFrame(calib_corr_map, columns=['f', 'xp', 'yp', 'k1', 'k2', 'k3', 'p1', 'p2', 'b1', 'b2'],
                                index=['f', 'xp', 'yp', 'k1', 'k2', 'k3', 'p1', 'p2', 'b1', 'b2'])

    ax = sns.heatmap(
        corr_map_dat,
        vmin=-1, vmax=1,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot=True
    )
    # ax.set_xticklabels(
    #     ax.get_xticklabels(),
    #     rotation=45,
    #     horizontalalignment='right'
    # )


if __name__ == '__main__':
    fig, ax = plt.subplots()

    # plot_radial_distortion(4.48293e-04, 6.71816e-07, 3.22896e-10, ax)
    # plot_tangential_distortion(- 0.0007, 0.0009, ax)
    # plot_diffscale_distortion(0.0002, 0.0004, ax)
    # plot_dr2r(1.47e-5, -3.97e-9, 1.86e-13, 1.41e-6, 1.71e-6, -4.78e-5, -3.76e-5)

    # figure, axis = plt.subplots(1, 2)
    # axis[0].set_title("Positive Radial Distortion")
    # plot_radial_distortion(4.48293e-04, 6.71816e-07, 3.22896e-10, axis[0])
    #
    # # For Cosine Function
    # axis[1].set_title("Negative Radial Distortion")
    # plot_radial_distortion(-4.48293e-04, -6.71816e-07, -3.22896e-10, axis[1])

    # figure, axis = plt.subplots(1, 2)
    # axis[0].set_title("Positive Tangential Distortion")
    # plot_tangential_distortion(0.0007, 0.0009, axis[0])
    #
    # axis[1].set_title("Negative Tangential Distortion")
    # plot_tangential_distortion(-0.0007, -0.0009, axis[1])

    # figure, axis = plt.subplots()
    # axis.set_title("Differential Scale Distortion")
    # plot_diffscale_distortion(-0.00002, -0.00004, axis)

    ################
    # plot_all_distortions(1.47e-5, -3.97e-9, 1.86e-13, 1.41e-6, 1.71e-6, -4.78e-5, -3.76e-5, ax)
    ################

    ################
    fig, ax = plt.subplots(1, 2)
    plot_before_after(3.77e-8, -4.37e-12, -3.667e-16, -1.657e-7, -2.345e-7, -7.162e-6, 1.911e-5, 'orig_samples.csv', 'v.csv',
                      ax, scale=15)
    ################

    ################
    # plot_calib_corr_map('sigmaX.csv', ax, num_images=40)
    ################
    # display
    plt.show()
