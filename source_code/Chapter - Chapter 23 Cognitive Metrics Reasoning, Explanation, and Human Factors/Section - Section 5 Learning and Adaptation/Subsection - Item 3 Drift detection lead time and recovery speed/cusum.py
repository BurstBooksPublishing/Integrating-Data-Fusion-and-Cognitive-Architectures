import numpy as np, time

def cusum_stream_detector(stream, nu=0.0, h=8.0):
    S = 0.0
    alarms = []
    t = 0
    for x in stream:
        S = max(0.0, S + (x - nu))
        if S > h:
            alarms.append((t, S))
            S = 0.0  # reset after alarm
        t += 1
    return alarms

# simulate: baseline, onset at t=200, mitigation at t=350, recover at 420
np.random.seed(0)
baseline = np.random.normal(0,1,200)
drift = np.random.normal(2.5,1,220)  # shifted mean
post = np.random.normal(0,1,80)
stream = np.concatenate([baseline, drift, post])
alarms = cusum_stream_detector(stream, nu=0.5, h=10.0)  # tuned params

# compute lead and recovery times from timestamps (example labels)
t_onset = 200; t_mitigate = 350; t_recover = 420
t_detect = alarms[0][0] if alarms else None
lead_time = None if t_detect is None else (t_detect - t_onset)
recovery_time = t_recover - t_mitigate
print("alarm timestamps:", alarms)
print("lead_time:", lead_time, "recovery_time:", recovery_time)  # seconds/steps