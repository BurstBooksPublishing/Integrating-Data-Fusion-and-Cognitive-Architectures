#!/usr/bin/env python3
import time, subprocess, psutil

HEAVY = {"name":"model_heavy.onnx","E":0.25,"latency":0.02}  # J, s
LIGHT = {"name":"model_light.onnx","E":0.05,"latency":0.01}
P_TH = 10.0  # W thermal/power cap
P_BASE = 3.0  # W measured idle
P_AVAIL = P_TH - P_BASE

def read_temp():
    # try sysfs; fallback to psutil; temperature in C
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip())/1000.0
    except Exception:
        t = psutil.sensors_temperatures().get("cpu-thermal",[])
        return t[0].current if t else 50.0

def set_cpu_freq(freq_khz):
    # simple DVFS via cpufreq-set (requires permissions)
    subprocess.run(["cpufreq-set","-g","userspace"],check=False)
    subprocess.run(["cpufreq-set","-u",str(freq_khz)+"kHz"],check=False)

def choose_model(temp, recent_power):
    # conservative hysteresis and budget check
    if temp>75 or recent_power>P_AVAIL*0.9:
        return LIGHT
    if temp<65 and recent_power
\subsection{Item 2:  Latency SLOs, frame budgets, and memory footprints}
Building on the preceding discussion of embedded SoC, GPU/NPU and thermal ceilings, latency and memory targets must be expressed as system-level budgets that span perception, association, situation assessment, and cognitive reasoning. These budgets translate device constraints into operational SLOs, admission controls, and degradation policies used throughout the lifecycle.

Concept — what to budget and why
- Latency SLO: an end-to-end bound on time from sensor capture to action or rationale output, typically expressed as percentiles (p50/p95/p99). For a target frame rate $f$, the hard period is $T=1/f$ and the budget must satisfy timing and jitter allowances.
- Frame budget decomposition: partition SLO into stage budgets (L0→L4) and communication/queuing overheads to guarantee schedulability.
- Memory footprint: includes model weights, per-frame buffers, fixed-lag stores, track/graph structures, and OS/runtime overhead. Reserve for worst-case spikes and avoid swapping.

Process — deriving and validating budgets
1. Measure baseline: acquire p50/p95/p99 latencies for each stage on target hardware under representative thermal and co-scheduled loads.
2. Allocate stage budgets with margins: assign fractions $\alpha_i$ such that sum$(\alpha_i)=1-\delta$ where $\delta$ is safety margin for jitter and GC/OS interference.
3. Verify schedulability: ensure
   \begin{equation}[H]\label{eq:latency_sum}
   \sum_{i} \ell_i + \ell_{\text{comm}} + \ell_{\text{jitter}} \le T,
   \end{equation}
   where $\ell_i$ are stage p95 latencies.
4. Memory check: compute total footprint and compare to device cap $M_{\text{cap}}$:
   \begin{equation}[H]\label{eq:memory_total}
   M_{\text{total}} = \sum_j M_{\text{model},j} + B_{\text{frames}} + M_{\text{tracks}} + O \le M_{\text{cap}} - R,
   \end{equation}
   with $B_{\text{frames}}=$ frame\_size$\times$buffer\_depth, $O$ runtime/OS, and $R$ reserved headroom.
5. Apply mitigations if constraints violated: reduce frame rate, quantize/prune models, reduce buffer depth, enforce admission control for high-density scenes, or offload to cloud under bandwidth/assurance policies.
6. Validate across corners: thermal throttling, co-scheduling, and extreme track counts using HIL scenarios.

Example — numeric allocation for a 30 Hz pipeline
- Frame period $T=1/30 \approx 33.3\text{ ms}$. Target p95 SLO set to $33\text{ ms}$.
- Measuredstage p95s on target: perception $12\text{ ms}$, association $6\text{ ms}$, L2 $8\text{ ms}$, cognition $3\text{ ms}$, comms/jitter $2\text{ ms}$.
  These satisfy (1): $12+6+8+3+2=31\text{ ms}\le 33\text{ ms}$.
- Memory: raw frame 640×480×3 bytes $\approx0.92\text{ MB}$. Buffer depth 5 $\Rightarrow B_{\text{frames}}\approx4.6\text{ MB}$. Models: detector 25 MB, embedder 12 MB, reasoner 8 MB. Track store (100 tracks×512 bytes) = 51.2 KB. OS/runtime reserve 200 MB. Total $\approx245.6\text{ MB}$. If device cap $M_{\text{cap}}=256\text{ MB}$, reserve $R=16\text{ MB}$ violated; action required (quantize, reduce models, or increase cap).

Code — automated budget checker and simple remediation suggestion
\begin{lstlisting}[language=Python,caption={Budget checker for latency and memory on target device},label={lst:budget_checker}]
# simple, executable check; replace numbers with measured values
def check_budgets(frame_rate, stage_latencies_ms, comms_ms,
                  frame_shape, buffer_depth, model_sizes_mb,
                  runtime_overhead_mb, device_cap_mb, reserve_mb):
    T_ms = 1000.0 / frame_rate
    total_latency = sum(stage_latencies_ms) + comms_ms
    # memory calc
    frame_bytes = frame_shape[0]*frame_shape[1]*frame_shape[2]
    B_frames_mb = (frame_bytes*buffer_depth) / (1024**2)
    M_total_mb = sum(model_sizes_mb) + B_frames_mb + runtime_overhead_mb
    # results
    schedulable = total_latency <= T_ms
    mem_ok = M_total_mb <= (device_cap_mb - reserve_mb)
    return {
        "T_ms": T_ms, "total_latency_ms": total_latency, "schedulable": schedulable,
        "M_total_mb": M_total_mb, "mem_ok": mem_ok
    }

# example invocation (values from numeric example)
res = check_budgets(
    frame_rate=30,
    stage_latencies_ms=[12,6,8,3], # perception, assoc, L2, cognition
    comms_ms=2,
    frame_shape=(640,480,3),
    buffer_depth=5,
    model_sizes_mb=[25,12,8],
    runtime_overhead_mb=200,
    device_cap_mb=256,
    reserve_mb=16
)
print(res)  # quick diagnostic for CI/HIL gates