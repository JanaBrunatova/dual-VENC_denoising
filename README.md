This repository contain scripts and necessary files to reproduce synthetic data from the paper
**_Denoising of dual-VENC PC-MRI with large high/low VENC ratios_**

The dual-VENC unwrapping methods are adopted from a package by Miriam Löcke [1]; which can be found [here](https://git.web.rug.nl/p305235/Phase_Unwrapping_Comparison).

[1] Löcke M, Garay Labra JE,Franco P, Uribe S, Bertoglio C. A comparison of phase unwrapping methods in velocity-encoded MRI for aortic flows. Magn Reson Med.2023;90:2102-2115. doi: [10.1002/mrm.29767](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29767)

### Dependencies and packages
The code is written in Python3 and requires the following:
- FEniCS 2019.1.0 (see [here](https://fenicsproject.org/download/archive/))
- PyWavelets (see [here](https://github.com/PyWavelets/pywt)); we used version 1.4.1
- ruamel.yaml (see [here](https://pypi.org/project/ruamel.yaml/)); we used version 0.17.32

### Generate measurements
Synthetic measurements are created by using `gen_measurements.py` file with options specified in a separete yaml file.
For example, measurements on Plane 1 (aorta) are generated using:
```python
python3 gen_measurements.py input_files/measurements_aorta.yaml
```

The key options for generating measurements with noise are:
- signal-to-noise ratio (SNR): 9 or 12
- venc: 154 # cm/s
- phase_contrast line needs to be commented
- temporal undersampling:
  - aneurysm: 1 ($dt=30$), or 2 ($dt=60$)
  - aorta: 2 ($dt=30$), or 4 ($dt=60$)

Whereas for a noise-free measurement (which is the GroundTruth in our case)
- SNR: 'inf'
- phase_contrast: 100 # 100%
- venc must be commented


### Unwrapping and denoising
Similarly as for generating measurements, all options are specified in a separate yaml file. Three denoising methods were implemented, the ODV Correction, ODV Wavelet, and ODV Temporal Difference.
For example, ODV unwrapping and denoising of measurements on Plane 1 (aorta) is done by running:
```python
python3 unwrap.py input_files/unwrap_aorta.yaml
```
If multiple realizations are done and the option save_all_errors is True, mean errors and standard deviations are stored in a txt file.


### Specific options for synthetic measurements:

The high-VENC was set to approx. 120% of the velocity inside each domain:
- 154 cm/s (Plane 1, aorta),
- 120 cm/s (Plane 2, aorta),
- 172 cm/s (hexahedral mesh, aorta),
- 72 cm/s (plane, aneurysm).

Initial seeds were the following:
- 100 for $V_\text{H}$ = 120%,
- 10100 for $V_\text{L}$ = 60%,
- 20100 for $V_\text{L}$ = 30%.
