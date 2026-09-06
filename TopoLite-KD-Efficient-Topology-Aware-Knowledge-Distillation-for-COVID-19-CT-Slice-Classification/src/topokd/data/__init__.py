"""Dataset utilities with lazy loading of topology-dependent components."""

__all__ = ["CTTopologyDataset", "build_dataloaders", "load_manifest"]


def __getattr__(name):
    if name in __all__:
        from .datasets import CTTopologyDataset, build_dataloaders, load_manifest
        return {
            "CTTopologyDataset": CTTopologyDataset,
            "build_dataloaders": build_dataloaders,
            "load_manifest": load_manifest,
        }[name]
    raise AttributeError(name)
