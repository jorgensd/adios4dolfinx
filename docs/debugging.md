# Debugging

## Checkpoint Time Stamps and Floating Point Precision Issues

When writing and reading functions using `adios4dolfinx` within a time-stepping loop, users may encounter a `KeyError` when attempting to read a function at a specific time, even if the function was written at that seemingly identical time. This issue is typically caused by floating point precision errors introduced during the time integration process. See https://github.com/jorgensd/adios4dolfinx/issues/186 for more details.

### Solution

The most robust way to handle this is to check the actual time stamps saved in the checkpoint file and use the one that is closest to your desired time within a specified tolerance.You can retrieve the list of saved time stamps for a specific function name using the adios4dolfinx.read_timestamps function. Then, use $\texttt{numpy.argmin}$ along with $\texttt{numpy.abs}$ and $\texttt{numpy.isclose}$ to find the closest available time stamp and ensure it's within a reasonable tolerance of your target time. For example

```python
import numpy as np
# ... (setup and write operations from above) ...

# 1. Get the list of actual time stamps for the function "u"
time_stamps = adios4dolfinx.read_timestamps(
    checkpoint_fname, comm=comm, function_name="u"
)

target_time = 1.0
print(f"Available time stamps: {time_stamps}")

# 2. Find the index of the time stamp closest to the target_time
index = np.argmin(np.abs(time_stamps - target_time))
closest_time = time_stamps[index]

# 3. Check if the closest time is acceptably close to the target time
TOLERANCE = 1.0E-10 
if np.isclose(closest_time, target_time, atol=TOLERANCE):
    # 4. Read the function using the actual saved time stamp
    adios4dolfinx.read_function(checkpoint_fname, new_u, name="u", time=closest_time)
    print(f"Successfully read 'u' at time: {closest_time}")
else:
    raise ValueError(f"Target time {target_time} not found within tolerance {TOLERANCE}.")
```