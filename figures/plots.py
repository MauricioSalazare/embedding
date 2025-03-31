import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("MacOSX")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np

from figure_utils import set_figure_art

import matplotlib.gridspec as gridspec
from itertools import cycle, islice

matplotlib.rc('text', usetex=True)
set_figure_art()

#%%
#%% Load data
latent_vectors = pd.read_csv('../data/processed/latent_vectors.csv')
rlps_data = pd.read_csv('../data/processed/rlps_2023_data_clean.csv')


# This is required to maintain consistency between colors of the paper, for the clusters between figures
remap_cluster_dict = {0: 2, 1: 0, 2: 1, 3: 3}
latent_vectors['CLUSTER'] = latent_vectors['CLUSTER'].map(remap_cluster_dict)

#%%
print(latent_vectors["MODEL"].unique())
MODEL_NAMES = ['sphere', 'isomap', 'umap', 't-SNE', 'autoencoder', 'profile']

GEMEENTE = 'EINDHOVEN'
MONTH = 12
BOXID_BAD = "047.1279-1"
BOXID_GOOD = "047.1747-1"


##
MODEL = 'sphere'
z_frame = latent_vectors[(latent_vectors["GEMEENTE"] == GEMEENTE)
                         & (latent_vectors["MONTH"] == MONTH)
                         & (latent_vectors["MODEL"] == MODEL)]
z_frame = z_frame[["Z1", "Z2", "Z3", "CLUSTER"]]


bad_dali_z_frame = latent_vectors[(latent_vectors["GEMEENTE"] == GEMEENTE)
                                  & (latent_vectors["MONTH"] == MONTH)
                                  & (latent_vectors["MODEL"] == MODEL)
                                  & (latent_vectors["BOXID"] == BOXID_BAD)]

#%%
limits = {'sphere': {'min': -1.2,
                     'max': 1.2},
          'isomap': {'min': -20.0,
                     'max': 40.0},
          'umap': {'min': -8,
                   'max': 15.0},
          't-SNE': {'min': -20.0,
                    'max': 20.0},
          'vae': {'min': -12.0,
                  'max': 12.0},
          'autoencoder': {'min': -12.0,
                          'max': 12.0},
          }

titles = {'sphere': '(a)\nSphere',
          'isomap': '(b)\nIsomap',
          'umap': '(c)\nUMAP',
          't-SNE': '(d)\nt-SNE',
          'vae': '(e)\nVariational Autoencoder',
          'autoencoder': '(e)\nAutoencoder',
          }
super_title = "Municipality"
month_names = { 6: "July",
                7: "August",
               10: "October",
               12: "December"}
MARKER_SIZE = 2


n_cluster = 4
colors = list(islice(cycle(["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]), n_cluster))
centroids = dict()


# ================================================ START OF THE PLOT ===================================================
fig = plt.figure(figsize=(7.5, 6.0))
heights = [1, 1]
widths = [1, 1, 1]
outer = fig.add_gridspec(
    2, 3, wspace=0.2, hspace=0.0, left=0.1, bottom=0.00, right=0.9, top=0.92, width_ratios=widths, height_ratios=heights
)

outers = [outer[ii, jj] for ii in range(2) for jj in range(3)]

for ii, (outer, model) in enumerate(zip(outers, MODEL_NAMES)):
    print(f"interation: {ii}")
    if ii != 5:
        print(f"interation: {ii}")

        z_frame = latent_vectors[(latent_vectors["GEMEENTE"] == GEMEENTE)
                                 & (latent_vectors["MONTH"] == MONTH)
                                 & (latent_vectors["MODEL"] == model)]
        z_frame = z_frame[["Z1", "Z2", "Z3", "CLUSTER"]]

        bad_dali_z_frame = latent_vectors[(latent_vectors["GEMEENTE"] == GEMEENTE)
                                          & (latent_vectors["MONTH"] == MONTH)
                                          & (latent_vectors["MODEL"] == model)
                                          & (latent_vectors["BOXID"] == BOXID_BAD)]

        good_dali_z_frame = latent_vectors[(latent_vectors["GEMEENTE"] == GEMEENTE)
                                          & (latent_vectors["MONTH"] == MONTH)
                                          & (latent_vectors["MODEL"] == model)
                                          & (latent_vectors["BOXID"] == BOXID_GOOD)]

        height_ratios = [0.8,0.8, 1]
        width_ratios = [1,1,1]
        sub_1 = gridspec.GridSpecFromSubplotSpec(3, 3,
                                                 subplot_spec=outer, wspace=0.3, hspace=-0.3,
                                                 height_ratios=height_ratios, width_ratios=width_ratios)

        ax_a = plt.subplot(sub_1[0:2, 0:3], projection="3d")
        ax_a_0 = plt.subplot(sub_1[2, 0])
        ax_a_1 = plt.subplot(sub_1[2, 1])
        ax_a_2 = plt.subplot(sub_1[2, 2])

        for k, color in zip(range(n_cluster), colors):
            idx = z_frame["CLUSTER"] == k
            z_frame_filter = z_frame[idx]

            ax_a.scatter(
                z_frame_filter['Z1'].values,  # x
                z_frame_filter['Z2'].values,  # y
                z_frame_filter['Z3'].values,  # z
                marker="o",
                c=color,
                label=f"Cluster {k + 1} (C{k + 1})",
                edgecolors=color,
                s=MARKER_SIZE,
                zorder=3,
            )

        ax_a.scatter(
            bad_dali_z_frame['Z1'].values,  # x
            bad_dali_z_frame['Z2'].values,  # y
            bad_dali_z_frame['Z3'].values,  # z
            marker="o",
            c='purple',
            edgecolors='purple',
            s=MARKER_SIZE*20,
            zorder=3,
        )

        ax_a.scatter(
            good_dali_z_frame['Z1'].values,  # x
            good_dali_z_frame['Z2'].values,  # y
            good_dali_z_frame['Z3'].values,  # z
            marker="s",
            c='black',
            edgecolors=None,
            s=MARKER_SIZE * 20,
            zorder=3,
        )

        ax_a.view_init(elev=30, azim=60)

        ax_a.set_xticklabels([])
        ax_a.set_yticklabels([])
        ax_a.set_zticklabels([])
        ax_a.tick_params(axis="both", which="both", length=0)
        ax_a.set_xlabel("$z_1$", labelpad=-15, fontsize=8)
        ax_a.set_ylabel("$z_2$", labelpad=-15, fontsize=8)
        ax_a.set_zlabel("$z_3$", labelpad=-15, fontsize=8)
        ax_a.set_title(titles[model], fontsize=8, pad=-50)

        # # % ==================== SECOND ROW - 2D PROJECTIONS ====================================================================

        MAX_PRE = z_frame_filter[['Z1', 'Z2', 'Z3']].max().max()
        MIN_PRE = z_frame_filter[['Z1', 'Z2', 'Z3']].min().min()

        multiplier = 1.8
        if MAX_PRE > 1:
            multiplier = 2
        # if MAX_PRE > 10:
        #     multiplier = 10

        MAX = MAX_PRE * multiplier
        MIN = MIN_PRE * multiplier

        combinations = [('Z1', 'Z2'), ('Z2', 'Z3'), ('Z1', 'Z3')]
        combinations_labels = [('$z_1$', '$z_2$'), ('$z_2$', '$z_3$'), ('$z_1$', '$z_3$')]
        ax_subplots = [ax_a_0, ax_a_1, ax_a_2]


        for ii, (combination, combination_label, ax_subplot) in enumerate(zip(combinations, combinations_labels, ax_subplots)):
            for k in range(n_cluster):
                idx = z_frame["CLUSTER"] == k
                z_frame_filter = z_frame[idx]
                ax_subplot.scatter(
                    z_frame_filter[combination[0]].values,  # x
                    z_frame_filter[combination[1]].values,  # y
                    marker="o",
                    c=colors[k],
                    label=f"Cluster {k + 1} (C{k + 1})",
                    edgecolors=colors[k],
                    s=MARKER_SIZE,
                    zorder=k,
                )
                ax_subplot.scatter(
                    bad_dali_z_frame[combination[0]].values,  # x
                    bad_dali_z_frame[combination[1]].values,  # y
                    marker="o",
                    c='purple',
                    edgecolors='purple',
                    s=MARKER_SIZE*5,
                    zorder=k,
                )
                ax_subplot.scatter(
                    good_dali_z_frame[combination[0]].values,  # x
                    good_dali_z_frame[combination[1]].values,  # y
                    marker="s",
                    c='black',
                    edgecolors=None,
                    s=MARKER_SIZE * 5,
                    zorder=k,
                )

            # ax_a_0.plot(x_circle, y_circle, linewidth=0.4, color="grey", alpha=0.9, zorder=0)
            ax_subplot.set_xlim(limits[model]['min'], limits[model]['max'])
            ax_subplot.set_ylim(limits[model]['min'], limits[model]['max'])
            ax_subplot.set_box_aspect(1.0)
            # ax_a_0.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
            # ax_a_0.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            ax_subplot.set_xticklabels([])
            ax_subplot.set_yticklabels([])
            ax_subplot.tick_params(axis="both", which="both", length=0)
            ax_subplot.set_xlabel(combination_label[0], labelpad=-3, fontsize=6)
            ax_subplot.set_ylabel(combination_label[1], labelpad=-4, fontsize=6)
            # ax_subplot.set_title("(b)", pad=3, fontsize=5)

            ax_subplot.set_title(titles[model][:2] + str(ii+1) + ')', fontsize=6, pad=0)

    else:
        print("=======================")
        profile_bad = rlps_data[(rlps_data["MONTH"] == MONTH)
                                & (rlps_data["GEMEENTE"] == GEMEENTE)
                                & (rlps_data["BOXID"] == BOXID_BAD)].copy()
        profile_bad.drop(columns=["GEMEENTE", "BOXID", "MONTH", "CLUSTER"], inplace=True)

        profile_good = rlps_data[(rlps_data["MONTH"] == MONTH)
                                & (rlps_data["GEMEENTE"] == GEMEENTE)
                                & (rlps_data["BOXID"] == BOXID_GOOD)].copy()

        profile_good.drop(columns=["GEMEENTE", "BOXID", "MONTH", "CLUSTER"], inplace=True)

        sub_1 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer, wspace=0.3, hspace=0.5)
        ax_a = plt.subplot(sub_1[0, 0:3])
        ax_b = plt.subplot(sub_1[1, 0:3])

        ax_a.plot(profile_bad.values.flatten(), linewidth=0.8, color="purple", label="Inverted profile")
        ax_a.tick_params(axis='both', labelsize=5)
        # ax_a.set_xlabel("Time", labelpad=1, fontsize=7)
        ax_a.set_ylabel("Power [kW]", labelpad=1, fontsize=7)
        ax_a.set_title("(f)", fontsize=8, pad=5)
        ax_a.legend(fontsize=6, loc="upper right")


        ax_b.plot(profile_good.values.flatten(), linewidth=0.5, color="black", label="Normal profile")
        ax_b.tick_params(axis='both', labelsize=5)
        ax_b.set_xlabel("Time", labelpad=1, fontsize=7)
        ax_b.set_ylabel("Power [kW]", labelpad=1, fontsize=7)
        ax_b.set_title("(g)", fontsize=8, pad=5)
        ax_b.legend(fontsize=6, loc="upper right")

# Define legend elements
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markersize=6, label="Cluster 1"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='C1', markersize=6, label="Cluster 2"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='C2', markersize=6, label="Cluster 3"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='C3', markersize=6, label="Cluster 4"),

    Line2D([0], [0], marker='o', color='purple', markerfacecolor='purple', markersize=6, label="Inverted profile"),
    Line2D([0], [0], marker='s', color='black', markerfacecolor='black', markersize=6, label="Normal profile")
]

# Add a global legend to the bottom center
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=2,
    bbox_to_anchor=(0.78, 0.03),  # X: center, Y: just below the figure
    fontsize=7,
    frameon=False
)

plt.suptitle(f"Municipality 1 - {month_names[MONTH]}", y=0.99, fontsize=10, fontweight='bold')
plt.savefig(f"../data/figures/multiple_models_{month_names[MONTH]}.pdf")


