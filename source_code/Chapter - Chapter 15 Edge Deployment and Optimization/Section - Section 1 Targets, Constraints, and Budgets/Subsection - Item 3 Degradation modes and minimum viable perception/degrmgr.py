import time
# simple runtime monitor and mode switcher
def capability_score(cpu_frac, gpu_frac, temp_c, snr):
    # compute, sensor quality, thermal penalties
    fC = min(1.0, 1.0 - max(cpu_frac, gpu_frac))  # higher load -> lower fC
    fT = 1.0 if temp_c < 75 else max(0.2, 1.0 - (temp_c-75)/40)  # thermal headroom
    fS = max(0.0, min(1.0, snr/30.0))  # map SNR to [0,1]
    return fC * fT * fS

MODE_POLICY = [(0.7,'nominal'), (0.4,'lite'), (0.15,'mvp'), (0.0,'safe_stop')]

def select_mode(score):
    for threshold, mode in MODE_POLICY:
        if score >= threshold:
            return mode
    return 'safe_stop'

# main loop (example)
while True:
    cpu = read_cpu_util()      # 0..1
    gpu = read_gpu_util()      # 0..1
    temp = read_soc_temp()     # degC
    snr = read_sensor_snr()    # dB
    s = capability_score(cpu, gpu, temp, snr)
    mode = select_mode(s)
    publish_mode(mode)         # instruct perception pipeline
    publish_rationale({'score':s,'cpu':cpu,'gpu':gpu,'temp':temp,'snr':snr})
    time.sleep(0.2)            # loop period