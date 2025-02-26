import awkward as ak
import numpy as np


# Example arrays
array1 = np.array([1, 2, 3, 4])
array2 = np.array([4, 3, 2, 1, 6, 7, 8, 2, 1, 4])

# Sort array2 and get the sorted indices
sorted_indices = np.argsort(array1)
sorted_array1 = array1[sorted_indices]

# FIXME: handle nonexistent elements in array1

# Use searchsorted to find the indices of array2 elements in the sorted array1
indices_in_sorted = np.searchsorted(sorted_array1, array2)
is_in = np.isin(array2, sorted_array1)


# Map the sorted indices back to the original array2 indices
indices_in_array1 = sorted_indices[indices_in_sorted[is_in]]

print(f'{array1 = }')
print(f'{array2 = }')
print(f'{np.arange(len(array2))[is_in] = }')
print(f'{indices_in_array1 = }')


def get_fit_values(fit_data, channels) :
    sorted_indices = ak.argsort(fit_data.channel)
    sorted_data = fit_data.channel[sorted_indices]

    is_in = np.isin(channels, sorted_data)
    return ak.zip((ak.local_index(channels)[is_in],fit_data[sorted_indices[np.searchsorted(sorted_data, channels)[is_in]]]))


fit_data = ak.zip({'channel':array1})
channels = array2


print(f'{get_fit_values(fit_data,channels) = }')
get_fit_values(fit_data,channels).show()
