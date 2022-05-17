# Above all
Here is the code and data corresponding to the Physical Informed Deep Learning Reconstructs Missing Climate Information in the Antarctic article.
# Reanalysis data
In the **reanalysis data** folder, skin temperature data and wind speed data are in two different folders. The reanalysis data is the monthly average of the ERA-Interim from European Centre for Medium-Range Weather Forecasts (ECMWF) for 40 years from 1979 to 2018, with 1.5° latitude × 1.5° longitude resolution. The first 30 years are used as training data, and the next 10 years are used as test data. And the variables such as skin temperature (K, Kelvins) and surface wind speed (m/s, meter per second) are selected for experiments in Antarctic region (longitude (0°-360°), latitude (60°S-90°S)). 
# Station data
The **station data** folder contains real observations from 180 weather stations in the Antarctic region. These include variables such as temperature, wind speed, and wind direction.
# PI-RFR
**PI-RFR** folder contains the code for this method, including training, testing, mask generation and network structure.Requirements:
- Python >= 3.6
- PyTorch >= 1.6
- Numpy == 1.20.3
- NetCDF4 == 1.5.7
- h5py == 3.3.0
## Train & Test
To training or testing, use
```
python run.py
python run.py --test
```
