import pandas as pd
import numpy as np
from scipy.signal import correlate
# load exported parquet files (tracks and cpu metrics)
tracks = pd.read_parquet("tracks.parquet")  # columns: timestamp, track_id, x, y, cov_xx...
cpu = pd.read_parquet("cpu.parquet")        # columns: timestamp, node, cpu_pct
# normalize timestamps to UTC datetime
tracks['ts'] = pd.to_datetime(tracks['timestamp'], unit='s')
cpu['ts'] = pd.to_datetime(cpu['timestamp'], unit='s')
# resample to 100 ms uniform grid and aggregate
t0 = max(tracks['ts'].min(), cpu['ts'].min())
t1 = min(tracks['ts'].max(), cpu['ts'].max())
idx = pd.date_range(t0, t1, freq='100ms')
pos = tracks.set_index('ts').groupby('track_id')['x'].resample('100ms').mean().unstack(0).reindex(idx).interpolate()
cpu_ts = cpu.set_index('ts').groupby('node')['cpu_pct'].resample('100ms').mean().unstack(0).reindex(idx).interpolate()
# take one track and one node series
x = pos.iloc[:,0].fillna(method='ffill').to_numpy() - np.nanmean(pos.iloc[:,0])
y = cpu_ts['fusion_node'].fillna(method='ffill').to_numpy() - np.nanmean(cpu_ts['fusion_node'])
# compute cross-correlation and lag
corr = correlate(x, y, mode='full')
lags = np.arange(-len(x)+1, len(x))
lag_idx = lags[np.argmax(corr)]
lag_seconds = lag_idx * 0.1  # 0.1s bin
print(f"Estimated lag: {lag_seconds:.3f} s")
# simple residual anomaly: z-score on innovation (x - shifted prediction)
shift = int(max(0, lag_idx))
residual = x - np.roll(y, shift)
z = (residual - np.nanmean(residual)) / (np.nanstd(residual)+1e-9)
anomalies = np.where(np.abs(z) > 4)[0]  # flag >4 sigma events
# export anomaly table for dashboarding
anomaly_df = pd.DataFrame({'ts': idx[anomalies], 'track': pos.columns[0], 'z': z[anomalies]})
anomaly_df.to_parquet("anomalies_for_dashboard.parquet")  # artifact for Grafana/Loki ingestion