"""Datasets package.

Exposes the UAVid segmentation dataset so callers can import via
`from datasets import UAVIDDataset, NUM_CLASSES`
or `from datasets.UAVdatasets import UAVIDDataset`.
"""

from .UAVdatasets import UAVIDDataset, NUM_CLASSES, UAVID_CLASSES

__all__ = ["UAVIDDataset", "NUM_CLASSES", "UAVID_CLASSES"]
