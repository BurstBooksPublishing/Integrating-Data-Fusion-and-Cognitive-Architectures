import numpy as np
import pandas as pd

def fit_clock_model(ref_times, dev_times):
    # fit dev_time = a * ref_time + b  (a=1+drift, b=offset)
    A = np.vstack([ref_times, np.ones_like(ref_times)]).T
    a, b = np.linalg.lstsq(A, dev_times, rcond=None)[0]
    return a, b

def correct_timestamps(dev_df, a, b):
    # apply inverse mapping to get timestamps on ref clock
    dev_df['t_ref'] = (dev_df['t_device'] - b) / a
    return dev_df

# Example: calibration samples (collected periodically)
ref = np.array([1609459200.0, 1609459260.0, 1609459320.0])  # reference epoch times
dev = np.array([1609459199.8, 1609459259.85, 1609459320.12]) # device times
a,b = fit_clock_model(ref, dev)

# streaming correction (pandas DataFrame with device timestamps)
# dev_df = pd.DataFrame({'t_device': ..., 'value': ...})
# dev_df = correct_timestamps(dev_df, a, b)  # corrected times for fusion