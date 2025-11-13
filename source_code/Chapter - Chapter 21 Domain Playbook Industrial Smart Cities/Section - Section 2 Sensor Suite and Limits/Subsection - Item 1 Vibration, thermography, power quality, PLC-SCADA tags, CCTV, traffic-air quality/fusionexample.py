import time, math
import numpy as np

# Simulate vibration window (time domain)
def vibration_window(fs=5000, duration=0.5, fault=False):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    base = 0.02*np.sin(2*np.pi*120*t)  # normal tone
    if fault:
        base += 0.1*np.sin(2*np.pi*2400*t)  # high-frequency fault
    noise = 0.005*np.random.randn(len(t))
    return base + noise

# Simple FFT-based peak finder
def spectral_peak_score(x, fs, band=(2000,3000)):
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    idx = np.logical_and(freqs>=band[0], freqs<=band[1])
    peak = X[idx].max() if idx.any() else 0.0
    return peak / (np.linalg.norm(X)+1e-9)

# Thermography: ROI max temp (simulated)
def thermography_roi(max_temp=60.0, bias=0.0):
    return max_temp + bias + 0.5*np.random.randn()

# PLC/SCADA tag read (simulated)
def read_plc_tag():
    # tag state e.g., motor running flag and setpoint temperature
    return {"motor_running": True, "setpoint": 55.0}

# Fusion rule: combine modalities with weights and cognitive gate
def fused_anomaly_score(vib_score, temp, plc, thresholds):
    temp_anom = 1.0 if temp > thresholds["temp"] else 0.0
    plc_risk = 1.0 if not plc["motor_running"] else 0.0
    score = 0.6*vib_score + 0.3*temp_anom + 0.1*plc_risk
    return score

# Run loop (single cycle)
fs = 5000
vib = vibration_window(fs, fault=True)        # real read
vib_score = spectral_peak_score(vib, fs)      # L0 feature
temp = thermography_roi(bias=6.0)             # L0 feature
plc = read_plc_tag()                           # L0 tag

thresholds = {"temp": 58.0, "alert": 0.25}
score = fused_anomaly_score(vib_score, temp, plc, thresholds)

if score > thresholds["alert"]:
    alert = {"type":"asset_alert","score":float(score),
             "evidence":{"vib_peak":float(vib_score),"temp":float(temp)}}
    print("ALERT:", alert)  # replace with publish to cognition service
else:
    print("OK: score", score)