import warnings
import torch
from .util import ensure_array
import numpy as np

def _check_plt(loader, n_samples, instance_labels, model=None, device=None):
    import matplotlib.pyplot as plt
    img_size = 5

    fig = None
    n_rows = None

    def to_index(ns, rid, sid):
        index = 1 + rid * ns + sid
        return index

    for ii, (x, y) in enumerate(loader):
        if ii >= n_samples:
            break

        if model is None:
            pred = None
        else:
            pred = model(x if device is None else x.to(device))
            pred = ensure_array(pred)[0]

        # cast the data to array and remove the batch axis / choose first sample in batch
        x = ensure_array(x)[0]
        y = ensure_array(y)[0]

        print(f"Label shape: {y.shape}")
        print(f"Number of annotated pixels: {len(y[y != -1])} out of {y.size}")
        print(f"Number of annotated slices: {len(y[y != -1])} out of {y.size}")
        assert x.ndim == y.ndim

        if x.ndim == 4:  # 3d data (with channel axis)
            x_slice = np.s_[:, :, x.shape[3] // 2]
            y_slice = np.s_[:, x.shape[2] // 2, :]
            z_slice = np.s_[x.shape[1] // 2, :, :]

            # warnings.warn(f"3d input data is not yet supported, will only show slice {z_slice} / {x.shape[1]}")
            # x, y = x[:, z_slice], y[:, z_slice]
            print(f"3D data, showing slices at {(x.shape[3] // 2, x.shape[2] // 2, x.shape[1] // 2)} out of {x.shape}")
            if pred is not None:
                pred = pred[:, z_slice]
            n_projections = 3

        if x.ndim < 4:
            x = x[:, np.newaxis, ...]
            y = y[:, np.newaxis, ...]
            if pred is not None:
                pred = pred[:, np.newaxis, ...]

            n_projections = 1
            z_slice = np.s_
        if x.shape[0] > 1:
            warnings.warn(f"Multi-channel input data is not yet supported, will only show channel 0 / {x.shape[0]}")
        x = x[0]

        if pred is None:
            n_target_channels = y.shape[0]
        else:
            n_target_channels = pred.shape[0]
            y = y[:n_target_channels]
            assert y.shape[0] == n_target_channels

        if fig is None:
            n_rows = n_target_channels + 1 if pred is None else 2 * n_target_channels + 1
            fig = plt.figure(figsize=((n_samples  * n_projections) * img_size, n_rows * img_size))

        ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 0, ii * n_projections))
        ax.imshow(x[z_slice], interpolation="nearest", cmap="Greys_r", aspect="auto")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plots_kwargs = {"interpolation": "nearest", "cmap": "Greys_r", "aspect": "auto", "vmin": y.min(), "vmax": y.max() }

        if n_projections == 3:
            ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 0, ii * n_projections + 1))
            ax.imshow(x[y_slice], interpolation="nearest", cmap="Greys_r", aspect="auto")
            ax.set_xlabel("x")
            ax.set_ylabel("z")

            ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 0, ii * n_projections + 2))
            ax.imshow(x[x_slice], interpolation="nearest", cmap="Greys_r", aspect="auto")
            ax.set_xlabel("y")
            ax.set_ylabel("z")
        for chan in range(n_target_channels):
            ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 1 + chan, ii * n_projections))
            if instance_labels:
                ax.imshow(y[chan][z_slice].astype("uint32"), interpolation="nearest", aspect="auto")
            else:
                ax.imshow(y[chan][z_slice], **plots_kwargs)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            if n_projections == 3:
                ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 1 + chan, ii * n_projections + 1))
                if instance_labels:
                    ax.imshow(y[chan][y_slice].astype("uint32"), interpolation="nearest", aspect="auto")
                else:
                    ax.imshow(y[chan][y_slice], **plots_kwargs)
                ax.set_xlabel("x")
                ax.set_ylabel("z")

                ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 1 + chan, ii * n_projections + 2))
                if instance_labels:
                    ax.imshow(y[chan][x_slice].astype("uint32"), interpolation="nearest", aspect="auto")
                else:
                    ax.imshow(y[chan][x_slice], **plots_kwargs)
                ax.set_xlabel("y")
                ax.set_ylabel("z")

        if pred is not None:
            for chan in range(n_target_channels):
                ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 1 + n_target_channels + chan, ii * n_projections))
                ax.imshow(pred[chan][z_slice], **plots_kwargs)

                if n_projections == 3:
                    ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 1 + chan, ii * n_projections + 1))
                    ax.imshow(pred[chan][y_slice], **plots_kwargs)
                    ax.set_xlabel("x")
                    ax.set_ylabel("z")

                    ax = fig.add_subplot(n_rows, n_samples * n_projections, to_index(n_samples * n_projections, 1 + chan, ii * n_projections + 2))
                    ax.imshow(pred[chan][x_slice], **plots_kwargs)
                    ax.set_xlabel("y")
                    ax.set_ylabel("z")

    plt.show()
    return fig


def _check_napari(loader, n_samples, instance_labels, model=None, device=None):
    import napari

    for ii, (x, y) in enumerate(loader):
        if ii >= n_samples:
            break

        if model is None:
            pred = None
        else:
            pred = model(x if device is None else x.to(device))
            pred = ensure_array(pred).squeeze(0)

        x = ensure_array(x).squeeze(0)
        y = ensure_array(y).squeeze(0)

        v = napari.Viewer()
        v.add_image(x)
        if instance_labels:
            v.add_labels(y.astype("uint32"))
        else:
            v.add_image(y)
        if pred is not None:
            v.add_image(pred)
        napari.run()


def check_trainer(trainer, n_samples, instance_labels=False, split="val", loader=None, plt=False):
    if loader is None:
        assert split in ("val", "train")
        loader = trainer.val_loader
    with torch.no_grad():
        model = trainer.model
        model.eval()
        if plt:
            _check_plt(loader, n_samples, instance_labels, model=model, device=trainer.device)
        else:
            _check_napari(loader, n_samples, instance_labels, model=model, device=trainer.device)


def check_loader(loader, n_samples, instance_labels=False, plt=False):
    if plt:
        _check_plt(loader, n_samples, instance_labels)
    else:
        _check_napari(loader, n_samples, instance_labels)
