import pytorch_lightning as pl
from torch.utils.data import DataLoader

from omegaconf import DictConfig


class SAEBaseDataModule(pl.LightningDataModule):
    """Builds train/val/test loaders from Hydra ``dataset`` config."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        from saebase.registry import get_dataset_plugin

        name = str(self.cfg.dataset.name)
        plugin = get_dataset_plugin(name)
        plugin.assert_task_supported(self.cfg)
        plugin.setup(self, self.cfg, stage)

    def train_dataloader(self):
        ds = self.cfg.dataset
        return DataLoader(
            self.train_set,
            batch_size=ds.get("batch_size", 8),
            shuffle=True,
            num_workers=ds.get("num_workers", 0),
            pin_memory=ds.get("pin_memory", False),
        )

    def val_dataloader(self):
        ds = self.cfg.dataset
        return DataLoader(
            self.val_set,
            batch_size=ds.get("batch_size", 8),
            shuffle=False,
            num_workers=ds.get("num_workers", 0),
            pin_memory=ds.get("pin_memory", False),
        )

    def test_dataloader(self):
        ds = self.cfg.dataset
        return DataLoader(
            self.test_set,
            batch_size=ds.get("batch_size", 8),
            shuffle=False,
            num_workers=ds.get("num_workers", 0),
            pin_memory=ds.get("pin_memory", False),
        )

    def eval_split_note(self) -> str:
        """Human-readable note for summary.log about val/test splits."""
        from saebase.registry import get_dataset_plugin

        plugin = get_dataset_plugin(str(self.cfg.dataset.name))
        return plugin.eval_split_note(self)
