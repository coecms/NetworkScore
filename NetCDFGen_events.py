import netCDF4 as nc
import random
import numpy as np

fn = 'Event_severity.nc'
ds = nc.Dataset(fn, 'w', format='NETCDF4')

# Number of Timesteps:
Number_Of_Ticks = 10000
# Event severity:
event_lev = [1, 3, 5]
# Weights for each Event severity level:
# 1 - 60%, 3 - 30%, 5 - 10%
weights = [60, 30, 10]

# Create the following dimensions
T = ds.createDimension('time', Number_Of_Ticks)
lat = ds.createDimension('lat', 100)
lon = ds.createDimension('lon', 100)

# Create the following variablea and assign them to their respective dimensions
Ts = ds.createVariable('time', 'f4', ('time',))
lats = ds.createVariable('lat', 'f4', ('lat',))
lons = ds.createVariable('lon', 'f4', ('lon',))
events = ds.createVariable('events', 'f4', ('time', 'lat', 'lon',), chunksizes=(1, 100, 100))

# Create the following attributes
events.units = 'Scaler'
lats.units = 'Degrees'
lons.units = 'Degrees'
Ts.units = 'Days'

# Add data to the dimensions
lats[:] = np.arange(0, 100, 1.0)
lons[:] = np.arange(0, 100, 1.0)
Ts[:] = np.arange(1, Number_Of_Ticks+1, 1.0)

# Matrix of zeros to match lat,lon grid
# Using this matrix to store the event severity levels
Y = np.zeros((100, 100))

# Iterates over the timesteps
for i in range(Number_Of_Ticks):
    print('Normal', i)

    # Matrix of zeros to match lat,lon grid
    X = np.zeros((100, 100))

    # Generate random coordinates equal to the timestep number
    coordinates = np.random.randint(0, 100, (i + 1 , 2))
    # Set these coordinates to be 1
    X[coordinates[:, 0], coordinates[:, 1]] = 1
    # Set the coordinates to be 1,3,5
    Y[coordinates[:, 0], coordinates[:, 1]] = random.choices(event_lev, weights=weights)[0]

    # Find all the coordinates of the non-zero values in the Y matrix
    non_zero_indices = np.nonzero(Y)
    # Set these coordinates in X to be 1
    X[non_zero_indices] = 1
    # Reduce the non-zero values by 1 each timestep
    Y[non_zero_indices] = Y[non_zero_indices] - 1

    # Puts the actual values into the NetCDF
    events[i, :, :] = X

# Close and save the NetCDF file
ds.close()