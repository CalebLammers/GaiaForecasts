# [Gaia Exoplanet Forecasts](https://arxiv.org/abs/????.?????)
This repository contains code and supplementary material for the Gaia forecasts presented in Lammers & Winn submitted (https://arxiv.org/abs/????.?????). Scripts are included to reproduce the semi-analytical calculation, generate and fit simulated Gaia astrometry, and make the figures that appear in the paper.

![](parameter_space.png)

# Mock DR4 & DR5 exoplanet catalogs
Our mock catalogs are included as '.csv' files and can be accessed using the code below.

```python
import pandas as pd

dataframe = pd.read_csv('DR4_mock_exoplanet_catalog.csv', encoding='utf-8') # 'DR4' or 'DR5'

print(dataframe.shape)
# >>> (7545, 35)

print(dataframe.keys()) # 35 quantities in catalog
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
The code in 'Figure notebooks' illustrates how the catalogs can be used to reproduce the figures that appear in the paper. Stars in the mock catalog are real Gaia sources, so any desired parameters that were not included in the catalog can be retrieved using the Gaia source IDs (see 'CMD for planet hosting stars and binaries (Fig 15).ipynb').

# Mock DR4 & DR5 planet impostor binary catalogs

The mock catalogs of planet-impostor binaries can be loaded analogously to the exoplanet catalogs.

```python
import pandas as pd

dataframe = pd.read_csv('DR4_mock_planet_impostor_catalog.csv', encoding='utf-8') # 'DR4' or 'DR5'

print(dataframe.shape)
# >>> (1151, 38)

print(dataframe.keys()) # 38 quantities in catalog
# >>> Index(['Binary source ID', 'True distance [pc]', 'True RA [deg]',
#       'True Dec [deg]', 'Primary stellar mass [M_\odot]',
#       'Secondary stellar mass [M_\odot]', 'Apparent stellar mass [M_\odot]',
#       'Primary G-band mag', 'Secondary G-band mag', 'Apparent G-band mag',
#       'True period [days]', 'True inclination [deg]', 'True eccentricity',
#       'True omega [deg]', 'True Omega [deg]', 'True T_peri [days]',
#       'Best-fit planet mass [M_J]', 'Best-fit period [days]',
#       'Best-fit inclination [deg]', 'Best-fit eccentricity',
#       'Best-fit omega [deg]', 'Best-fit Omega [deg]',
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
The true parameters for the stellar companions are included, along with the 'apparent' values that would have been inferred if the sources were erroneously interpreted as single stars hosting planets.

# Creating custom mock catalogs
Results for all orbit fits are included in a master file, allowing users to create mock catalogs with different detection criteria. The code below illustrates how to load the orbit fits and apply the detection criteria used to make the fiducial catalogs.

```python
import pandas as pd

dataframe = pd.read_csv('DR4_master_orbital_fits.csv', encoding='utf-8') # 'DR4' or 'DR5'

print(dataframe.shape)
# >>> (29129, 35)

planet_masses = np.array(dataframe['Best-fit planet mass [M_J]'])
Porb_84th = np.array(dataframe['MCMC period 84th [days]'])
Porb_16th = np.array(dataframe['MCMC period 16th [days]'])
detection_criteria = ((planet_masses < 13.0) & (Porb_84th/Porb_16th < 1.5))
print(len(planet_masses[detection_criteria])) # as in exoplanet catalog
# >>> 7545
```
We restricted our attention to planets with \Delta \chi^2 > 50 and an orbital period < 7 years for DR4 (< 14 years for DR5). To modify these choices, one must re-run the experiment ('PYTHON FILE NAME'), which requires computational resources or patience.

# Contact
Feel free to contact me at caleb [dot] lammers [at] princeton [dot] edu if you have questions/comments!
