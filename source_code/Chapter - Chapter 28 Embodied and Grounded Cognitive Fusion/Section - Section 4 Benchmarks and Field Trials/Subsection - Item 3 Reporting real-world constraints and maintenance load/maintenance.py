import pandas as pd
import numpy as np

# telemetry: DataFrame with columns: asset_id, event_type, count, period_hours, service_hours_per_event
def predict_maintenance_load(telemetry: pd.DataFrame, horizon_hours: float):
    # aggregate rates per asset and event
    agg = telemetry.groupby(['asset_id','event_type']).agg({
        'count':'sum', 'period_hours':'sum', 'service_hours_per_event':'mean'
    }).reset_index()
    # compute lambda (events per hour) and predicted hours per event type
    agg['lambda'] = agg['count'] / agg['period_hours']  # events/hour
    agg['pred_hours'] = agg['lambda'] * agg['service_hours_per_event'] * horizon_hours
    # sum across event types to get per-asset forecast
    asset_load = agg.groupby('asset_id')['pred_hours'].sum().reset_index()
    asset_load = asset_load.sort_values('pred_hours', ascending=False)
    # MTBF estimate per asset (simple inverse of total lambda)
    lambda_tot = agg.groupby('asset_id')['lambda'].sum().reset_index(name='lambda_tot')
    asset_load = asset_load.merge(lambda_tot, on='asset_id')
    asset_load['mtbf_hours'] = np.where(asset_load['lambda_tot']>0,
                                       1.0 / asset_load['lambda_tot'], np.inf)
    return asset_load  # DataFrame: asset_id, pred_hours, lambda_tot, mtbf_hours

# Example usage (mock data) omitted for brevity