from typing import Iterable


def create_mouse_move_handler(fig, axes_list):
    if not isinstance(axes_list, Iterable):
        axes_list_fn = [axes_list]
    else:
        axes_list_fn = axes_list

    def on_mouse_move(event):
        for ax in axes_list_fn:
            if event.button == 1 and event.inaxes == ax:  # Left mouse button
                elev, azim = ax.elev, ax.azim  # Get current rotation angles
                ax.view_init(elev=elev, azim=azim)  # Lock roll (Z-axis)
        fig.canvas.draw_idle()  # Refresh all plots
    return on_mouse_move  # Return the handler function


def plot_3D(data, labels, title, ax):
    if labels is None:
        color = 'grey'
        cmap = None
    else:
        color = labels
        cmap = 'tab10'

    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=cmap, alpha=0.7, s=2)
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')

    return scatter
