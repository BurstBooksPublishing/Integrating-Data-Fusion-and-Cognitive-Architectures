import pandas as pd
import numpy as np

# fused_events: DataFrame with columns arrival_time, departure_time, lane_id, signal_id,
# preempt_requested (bool), preempt_granted_time (na if not granted)
def compute_kpis(fused_events, interval='5min'):
    df = fused_events.copy()
    df['arrival_time'] = pd.to_datetime(df['arrival_time'])
    df['departure_time'] = pd.to_datetime(df['departure_time'])
    df['delay'] = (df['departure_time'] - df['arrival_time']).dt.total_seconds()
    # interval aggregation
    df.set_index('arrival_time', inplace=True)
    agg = df.resample(interval).agg({
        'delay': ['mean','count'],                   # average delay and flow
        'lane_id': 'nunique'                         # lane diversity as health indicator
    })
    agg.columns = ['delay_mean','flow_count','lane_diversity']
    # Little's law queue estimate per interval
    agg['arrival_rate'] = agg['flow_count'] / (pd.to_timedelta(interval).total_seconds())
    agg['queue_est'] = agg['arrival_rate'] * agg['delay_mean']  # L = lambda * W
    # preemption success
    pre = df[df['preempt_requested']==True].copy()
    pre['granted'] = pre['preempt_granted_time'].notna()
    pre_rate = pre.resample(interval)['granted'].mean().rename('preempt_success_rate')
    # combine
    result = agg.join(pre_rate).fillna({'preempt_success_rate':0.0})
    return result

# Example usage: kpis = compute_kpis(fused_events_df)