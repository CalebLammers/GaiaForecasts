# [Gaia Exoplanet Forecasts](https://arxiv.org/abs/????.?????)
This repository contains code and supplementary material for the Gaia forecasts presented in Lammers & Winn submitted (https://arxiv.org/abs/????.?????). Scripts are included to reproduce our semi-analytical calculation, generate and fit simulated Gaia astrometry, and make the figures that appear in the paper.

Prerequisites for re-creating catalogs: numpy, matplotlib, os, ctypes, healpy, emcee, and gaiamock

# Mock DR4 & DR5 exoplanet catalogs
Our mock catalogs are included as '.csv' files and can be accessed using the code below.

```python
import pandas as pd

dataframe = pd.read_csv('DR4_mock_exoplanet_catalog.csv', encoding='utf-8') 

print(dataframe.shape)
# >>> (7545, 35)

print(dataframe.keys())
# >>> Index(['Gaia source IDs', 'True distance [pc]', 'True RA [deg]',
#       'True Dec [deg]', 'Stellar mass [M_\odot]', 'G-band mag',
#       'True planet mass [M_J]', 'True period [days]',
#       'True inclination [deg]', 'True eccentricity', 'True omega [deg]',
#       'True Omega [deg]', 'True T_peri [days]', 'Best-fit planet mass [M_J]',
#       'Best-fit period [days]', 'Best-fit inclination [deg]',
#       'Best-fit eccentricity', 'Best-fit omega [deg]', 'Best-fit Omega [deg]',
#       'Best-fit T_peri [days]', 'MCMC distance 16th [pc]',
#       'MCMC distance 50th [pc]', 'MCMC distance 84th [pc]',
#       'MCMC period 16th [days]', 'MCMC period 50th [days]',
#       'MCMC period 84th [days]', 'MCMC planet mass 16th [M_J]',
#       'MCMC planet mass 50th [M_J]', 'MCMC planet mass 84th [M_J]',
#       'MCMC eccentricity 16th', 'MCMC eccentricity 50th',
#       'MCMC eccentricity 84th', 'MCMC inclination 16th [deg]',
#       'MCMC inclination 50th [deg]', 'MCMC inclination 84th [deg]'],
#      dtype='object')
```

The stars in the mock catalog are real Gaia sources, so any desired parameters that were not included in the catalog can be retrieved using the Gaia source IDs. The mock catalogs of planet-impostor binaries ('.csv' and '.csv') can be loaded analogously.

![](parameter_space.png)

# Creating custom mock catalogs
The orbit fits for all planets are included as '.csv' files so that users can create mock catalogs with different detection criteria. The code below illustrates how to load the orbit fits and apply the detection criteria used to make our fiducial catalogs.

```python
import SOMETHING

LOAD CATALOGS

print(len(SOMETHING))
# >>> 12345566
```

# Contact
Feel free to contact me at caleb [dot] lammers [at] princeton [dot] edu if you have questions/comments.
