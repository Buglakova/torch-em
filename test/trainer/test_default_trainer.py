import os
import unittest
from shutil import rmtree

import h5py
import numpy as np
import torch

from torch_em import default_segmentation_loader
from torch_em.loss import DiceLoss
from torch_em.model import UNet2d


class TestDefaultTrainer(unittest.TestCase):
    data_path = "./data.h5"
    checkpoint_folder = "./checkpoints"
    log_folder = "./logs"
    name = "test"

    def setUp(self):
        shape = (8, 128, 128)
        chunks = (1, 128, 128)
        with h5py.File(self.data_path, "w") as f:
            f.create_dataset("raw", data=np.random.rand(*shape), chunks=chunks)
            f.create_dataset("labels", data=np.random.randint(0, 32, size=shape), chunks=chunks)

    def tearDown(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        if os.path.exists(self.checkpoint_folder):
            rmtree(self.checkpoint_folder)
        if os.path.exists(self.log_folder):
            rmtree(self.log_folder)

    def _get_kwargs(self):
        loader = default_segmentation_loader(
            raw_paths=self.data_path, raw_key="raw",
            label_paths=self.data_path, label_key="labels",
            batch_size=1, patch_shape=(1, 128, 128), ndim=2
        )
        model = UNet2d(in_channels=1, out_channels=1,
                       depth=2, initial_features=4)
        kwargs = {
            "name": self.name,
            "train_loader": loader,
            "val_loader": loader,
            "model": model,
            "loss": DiceLoss(),
            "metric": DiceLoss(),
            "optimizer": torch.optim.Adam(model.parameters(), lr=1e-5),
            "device": torch.device("cpu"),
            "mixed_precision": False
        }
        return kwargs

    def test_fit(self):
        from torch_em.trainer import DefaultTrainer
        trainer = DefaultTrainer(**self._get_kwargs())
        trainer.fit(10)

        save_folder = os.path.join(self.checkpoint_folder, self.name)
        self.assertTrue(os.path.exists(save_folder))
        best_ckpt = os.path.join(save_folder, "best.pt")
        self.assertTrue(os.path.exists(best_ckpt))
        latest_ckpt = os.path.join(save_folder, "latest.pt")
        self.assertTrue(os.path.exists(latest_ckpt))
        self.assertEqual(trainer.iteration, 10)

        trainer.fit(2)
        self.assertEqual(trainer.iteration, 12)

        trainer = DefaultTrainer(**self._get_kwargs())
        trainer.fit(8, load_from_checkpoint="latest")
        self.assertEqual(trainer.iteration, 20)

    def test_from_checkpoint(self):
        from torch_em.trainer import DefaultTrainer
        trainer = DefaultTrainer(**self._get_kwargs())
        trainer.fit(10)

        trainer2 = DefaultTrainer.from_checkpoint(
            os.path.join(self.checkpoint_folder, self.name),
            name="latest"
        )
        self.assertEqual(trainer.iteration, trainer2.iteration)
        self.assertEqual(trainer.model.depth, trainer2.model.depth)

        # make sure that the optimizer was loaded properly
        lr1 = [pm["lr"] for pm in trainer.optimizer.param_groups][0]
        lr2 = [pm["lr"] for pm in trainer2.optimizer.param_groups][0]
        self.assertEqual(lr1, lr2)

        trainer2.fit(10)
        self.assertEqual(trainer2.iteration, 20)


if __name__ == "__main__":
    unittest.main()
