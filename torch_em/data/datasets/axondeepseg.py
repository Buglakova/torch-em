import os
from glob import glob
from shutil import rmtree

import imageio
import h5py
import numpy as np
import torch_em
from .util import download_source, unzip, update_kwargs

URLS = {
    "sem": "https://github.com/axondeepseg/data_axondeepseg_sem/archive/refs/heads/master.zip",
    "tem": ""
}
CHECKSUMS = {
    "sem": "d334cbacf548f78ce8dd4a597bf86b884bd15a47a230a0ccc46e1ffa94d58426",
    "tem": "a"
}


def _preprocess_sem_data(out_path):
    # preprocess the data to get it to a better data format
    data_root = os.path.join(out_path, "data_axondeepseg_sem-master")
    assert os.path.exists(data_root)

    # get the raw data paths
    raw_folders = glob(os.path.join(data_root, "sub-rat*"))
    raw_folders.sort()
    raw_paths = []
    for folder in raw_folders:
        paths = glob(os.path.join(folder, "micr", "*.png"))
        assert len(paths) == 1
        raw_paths.append(paths[0])

    # get the label paths
    label_folders = glob(os.path.join(
        data_root, "derivatives", "labels", "sub-rat*"
    ))
    label_folders.sort()
    label_paths = []
    for folder in label_folders:
        paths = glob(os.path.join(folder, "micr", "*axonmyelin-manual.png"))
        assert len(paths) == 1
        label_paths.append(paths[0])
    assert len(raw_paths) == len(label_paths)

    # process raw data and labels
    for i, (rp, lp) in enumerate(zip(raw_paths, label_paths)):
        outp = os.path.join(out_path, f"sem_data_{i}.h5")
        with h5py.File(outp, "w") as f:

            # raw data: invert to match tem em intensities
            raw = imageio.imread(rp)
            assert np.dtype(raw.dtype) == np.dtype("uint8")
            raw = 255 - raw
            f.create_dataset("raw", data=raw, compression="gzip")

            # labels: map from [0, 127, 255] -> [0, 1, 2]
            labels = imageio.imread(lp)
            label_vals = np.unique(labels)
            assert np.array_equal(label_vals, [0, 127, 255])
            labels[labels == 127] = 1
            labels[labels == 255] = 2
            f.create_dataset("labels", data=labels, compression="gzip")

    # clean up
    rmtree(data_root)


# TODO figure out what to do for the git annexed tem data
def _require_axondeepseg_data(path, name, download):

    # download and unzip the data
    url, checksum = URLS[name], CHECKSUMS[name]
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, name)
    if os.path.exists(out_path):
        return out_path
    tmp_path = os.path.join(path, f"{name}.zip")
    download_source(tmp_path, url, download, checksum=checksum)
    unzip(tmp_path, out_path, remove=True)

    if name == "sem":
        _preprocess_sem_data(out_path)

    return out_path


# add instance segmentation representations?
def get_axondeepseg_loader(path, name, download=False, to_one_hot=False, **kwargs):
    data_root = _require_axondeepseg_data(path, name, download)
    paths = glob(os.path.join(data_root, "*.h5"))

    if to_one_hot:
        # add transformation to go from [0, 1, 2] to one hot encoding
        label_transform = torch_em.transform.label.OneHotTransform(class_ids=[0, 1, 2])
        msg = "'one_hot' is set to True, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    raw_key, label_key = "raw", "labels"
    return torch_em.default_segmentation_loader(
        paths, raw_key, paths, label_key, **kwargs
    )