"""
make_upload.py — creates a representative ~15MB sample for sharing.
Keeps ALL fault rows (BROKEN + RECOVERING) + every 12th NORMAL row.
Run from the project root:  python data/make_upload.py
"""
import pandas as pd, os

src = "data/pump_sensor.csv"
if not os.path.exists(src):
    raise FileNotFoundError(f"{src} not found. Download from Kaggle first.")

print("Loading full dataset...")
df = pd.read_csv(src)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

faults = df[df["machine_status"] != "NORMAL"]
normal = df[df["machine_status"] == "NORMAL"].iloc[::12]
sample = pd.concat([faults, normal]).sort_values("timestamp").reset_index(drop=True)
out    = "data/pump_upload.csv"
sample.to_csv(out, index=False)

mb = os.path.getsize(out) / 1e6
print(f"\nOriginal  : {len(df):,} rows  ({os.path.getsize(src)/1e6:.0f} MB)")
print(f"Upload    : {len(sample):,} rows  ({mb:.1f} MB)")
print(sample["machine_status"].value_counts().to_string())
print(f"\nReady to upload: {out}")
