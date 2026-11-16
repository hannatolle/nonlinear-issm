"""Preprocessing for the Huang et al. (2023) iEEG SPES dataset (OpenNeuro ds004457).

mef_io and preprocess require pymef and mne, which are not in the camel environment;
the saved arrays are then read by src.datasets.huang.HuangDataloader without them.
"""
