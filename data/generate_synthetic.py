"""
Generates a synthetic pump_sensor.csv that mirrors the Kaggle dataset exactly:
  - 52 sensor columns (sensor_00 … sensor_51)
  - machine_status: NORMAL ~93%, BROKEN ~3%, RECOVERING ~4%
  - ~220,320 rows at 1-minute intervals
  - sensor_15 has ~65% missing (mirrors the real data)
  - Several sensors are highly correlated
Run this ONLY if you cannot download the real dataset from Kaggle.
"""
import numpy as np
import pandas as pd

RNG  = np.random.default_rng(42)
N    = 220_320
FREQ = "1min"

# ── Timestamps ────────────────────────────────────────────────────────────────
start = pd.Timestamp("2018-04-01 00:00:00")
timestamps = pd.date_range(start, periods=N, freq=FREQ)

# ── Machine status (class proportions from real data) ─────────────────────────
# Fault windows are contiguous blocks, not scattered
status = np.array(["NORMAL"] * N, dtype=object)

def add_fault_window(arr, start_idx, length, label):
    end_idx = min(start_idx + length, N)
    arr[start_idx:end_idx] = label
    return arr

# ~7 BROKEN events scattered through the timeline
broken_starts   = [18000, 45000, 80000, 115000, 150000, 185000, 210000]
broken_lengths  = [800,   600,   1000,  700,    900,    500,    400   ]
recover_lengths = [1200,  1000,  1500,  1100,   1300,   800,    600   ]

for bs, bl, rl in zip(broken_starts, broken_lengths, recover_lengths):
    status = add_fault_window(status, bs, bl, "BROKEN")
    status = add_fault_window(status, bs + bl, rl, "RECOVERING")

# ── Sensor signals ─────────────────────────────────────────────────────────────
sensor_data = {}

# Base "health" signal (drives correlated sensors)
base_signal = RNG.normal(50, 5, N)
# Add degradation during faults
fault_mask = status == "BROKEN"
rec_mask   = status == "RECOVERING"
base_signal[fault_mask] += RNG.normal(25, 8, fault_mask.sum())
base_signal[rec_mask]   += RNG.normal(10, 4, rec_mask.sum())

for i in range(52):
    noise_scale = RNG.uniform(0.5, 3.0)
    offset      = RNG.uniform(-20, 80)
    scale       = RNG.uniform(0.3, 2.5)
    
    # Group sensors into correlated clusters
    if i < 12:                      # cluster A — strongly correlated to base
        sig = offset + scale * base_signal + RNG.normal(0, noise_scale, N)
    elif i < 22:                    # cluster B — moderate correlation
        sig = offset + scale * base_signal * 0.6 + RNG.normal(0, noise_scale * 2, N)
    elif i < 32:                    # cluster C — weak correlation, different scale
        sig = offset + RNG.uniform(0.1, 0.5) * base_signal + RNG.normal(0, noise_scale * 4, N)
    else:                           # cluster D — largely independent
        sig = offset + RNG.normal(0, noise_scale * 3, N)
        sig[fault_mask] += RNG.normal(5, 2, fault_mask.sum())

    # Introduce missing values
    if i == 15:
        missing_mask = RNG.random(N) < 0.65   # sensor_15: ~65% missing (mirrors real data)
        sig[missing_mask] = np.nan
    elif i in [7, 23, 41]:
        missing_mask = RNG.random(N) < 0.12
        sig[missing_mask] = np.nan
    elif i in [3, 18, 35, 47]:
        missing_mask = RNG.random(N) < 0.03
        sig[missing_mask] = np.nan

    sensor_data[f"sensor_{i:02d}"] = sig.round(4)

# ── Build DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame(sensor_data)
df.insert(0, "timestamp", timestamps)
df["machine_status"] = status

# ── Save ─────────────────────────────────────────────────────────────────────
df.to_csv("data/pump_sensor.csv", index=False)

# ── Verify ───────────────────────────────────────────────────────────────────
vc = df["machine_status"].value_counts()
print(f"Synthetic dataset saved: {df.shape}")
print(f"  NORMAL    : {vc['NORMAL']:,}  ({vc['NORMAL']/len(df)*100:.1f}%)")
print(f"  BROKEN    : {vc['BROKEN']:,}  ({vc['BROKEN']/len(df)*100:.1f}%)")
print(f"  RECOVERING: {vc['RECOVERING']:,}  ({vc['RECOVERING']/len(df)*100:.1f}%)")
print(f"  Missing in sensor_15: {df['sensor_15'].isna().sum():,} ({df['sensor_15'].isna().mean()*100:.1f}%)")
