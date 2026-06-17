import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import seaborn as sns
import warnings, os, json
warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("outputs/figures", exist_ok=True)
PALETTE={"NORMAL":"#2E75B6","BROKEN":"#C00000","RECOVERING":"#ED7D31"}
sns.set_theme(style="whitegrid",font_scale=1.05)
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False})

print("="*65)
print("NOTEBOOK 01 — Exploratory Data Analysis")
print("="*65)

# 1. Load
print("\n[1/7] Loading dataset...")
df=pd.read_csv("data/pump_sensor.csv")
df["timestamp"]=pd.to_datetime(df["timestamp"])
df=df.sort_values("timestamp").reset_index(drop=True)
df.set_index("timestamp",inplace=True)
sensor_cols=[c for c in df.columns if c.startswith("sensor")]
duration=(df.index.max()-df.index.min()).days
print(f"  Rows       : {len(df):,}")
print(f"  Sensors    : {len(sensor_cols)}")
print(f"  Duration   : {duration} days ({df.index.min().date()} -> {df.index.max().date()})")
print(f"  Classes    : {sorted(df['machine_status'].unique())}")

# 2. Class distribution
print("\n[2/7] Class distribution...")
cc=df["machine_status"].value_counts()
cp=df["machine_status"].value_counts(normalize=True)*100
for s in ["NORMAL","BROKEN","RECOVERING"]:
    print(f"  {s:<12}: {cc[s]:>7,}  ({cp[s]:.1f}%)")
print(f"  Imbalance NORMAL:BROKEN = {cc['NORMAL']/cc['BROKEN']:.1f}:1")
print(f"  Naive accuracy = {cp['NORMAL']:.2f}%  [NOT a valid metric]")

fig,axes=plt.subplots(1,2,figsize=(13,5))
fig.suptitle("Figure 1 — Pump Operating State Distribution\n(Kaggle Pump Sensor Dataset — Northern Nigeria Borehole Proxy)",fontsize=12,fontweight="bold",y=1.01)
colors=[PALETTE[s] for s in cc.index]
bars=axes[0].bar(cc.index,cc.values,color=colors,edgecolor="white",linewidth=1.5,width=0.5)
for bar,val in zip(bars,cc.values):
    axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+cc.max()*0.01,f"{val:,}",ha="center",fontweight="bold",fontsize=11)
axes[0].set_title("Count per State",fontsize=11,fontweight="bold")
axes[0].set_xlabel("Machine Status"); axes[0].set_ylabel("Observations")
axes[0].set_ylim(0,cc.max()*1.12)
_,_,autos=axes[1].pie(cc.values,labels=cc.index,autopct="%1.1f%%",colors=colors,startangle=90,wedgeprops={"edgecolor":"white","linewidth":2},pctdistance=0.75)
[t.set_fontweight("bold") for t in autos]
axes[1].set_title("Proportion",fontsize=11,fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/figures/01_class_distribution.png",dpi=150,bbox_inches="tight"); plt.close()
print("  -> Figure 1 saved")

# 3. Missing values
print("\n[3/7] Missing value audit...")
mn=df[sensor_cols].isnull().sum(); mp=(mn/len(df))*100
to_drop=mp[mp>30].index.tolist()
print(f"  No missing  : {(mn==0).sum()} sensors")
print(f"  Some missing: {(mn>0).sum()} sensors")
print(f"  DROP (>30%) : {len(to_drop)} -> {to_drop}")
sm=mp[mp>0].sort_values(ascending=False)
if len(sm):
    fig,ax=plt.subplots(figsize=(12,4))
    bc=["#C00000" if p>30 else "#ED7D31" if p>10 else "#2E75B6" for p in sm.values]
    ax.bar(sm.index,sm.values,color=bc,edgecolor="white")
    ax.axhline(30,color="black",linestyle="--",lw=1.5,label="Drop threshold 30%")
    ax.axhline(10,color="#ED7D31",linestyle="--",lw=1.0,label="Caution 10%")
    ax.set_title("Figure 2 — Missing Values by Sensor Channel",fontsize=12,fontweight="bold",pad=10)
    ax.set_xlabel("Sensor"); ax.set_ylabel("Missing (%)")
    ax.legend(fontsize=10); plt.xticks(rotation=45,ha="right"); plt.tight_layout()
    plt.savefig("outputs/figures/02_missing_values.png",dpi=150,bbox_inches="tight"); plt.close()
    print("  -> Figure 2 saved")

# 4. Time series
print("\n[4/7] Time-series plot...")
gs=[s for s in mp[mp<5].index if s in sensor_cols][:4]
if len(gs)<2: gs=sensor_cols[:4]
bm=df["machine_status"]=="BROKEN"; rm=df["machine_status"]=="RECOVERING"
fig,axes=plt.subplots(len(gs),1,figsize=(15,3.2*len(gs)),sharex=True)
if len(gs)==1: axes=[axes]
for ax,s in zip(axes,gs):
    lo=df[s].min(); hi=df[s].max(); pad=(hi-lo)*0.08 if (hi-lo)>0 else 0.5
    ax.plot(df.index,df[s],lw=0.35,color="#2E75B6",alpha=0.85,label=s)
    ax.fill_between(df.index,lo-pad,hi+pad,where=bm,alpha=0.25,color="#C00000",label="BROKEN")
    ax.fill_between(df.index,lo-pad,hi+pad,where=rm,alpha=0.20,color="#ED7D31",label="RECOVERING")
    ax.set_ylabel(s,fontsize=9,fontweight="bold"); ax.set_ylim(lo-pad,hi+pad)
    ax.legend(loc="upper right",fontsize=7,ncol=3,framealpha=0.9)
axes[-1].set_xlabel("Timestamp",fontsize=10)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=30,ha="right")
fig.suptitle("Figure 3 — Sensor Signals with Fault Period Highlighting",fontsize=12,fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/figures/03_timeseries_faults.png",dpi=150,bbox_inches="tight"); plt.close()
print("  -> Figure 3 saved")

# 5. Box plots
print("\n[5/7] Box plots by class...")
sb=gs[:3]; co=["NORMAL","BROKEN","RECOVERING"]
fig,axes=plt.subplots(1,len(sb),figsize=(14,5))
if len(sb)==1: axes=[axes]
for ax,s in zip(axes,sb):
    data=[df[df["machine_status"]==x][s].dropna().values for x in co]
    bp=ax.boxplot(data,labels=co,patch_artist=True,notch=False,showfliers=False,medianprops=dict(color="white",linewidth=2))
    for patch,lbl in zip(bp["boxes"],co): patch.set_facecolor(PALETTE[lbl]); patch.set_alpha(0.80)
    ax.set_title(s,fontweight="bold",fontsize=10); ax.set_ylabel("Sensor Value",fontsize=9); ax.tick_params(axis="x",labelsize=8)
fig.suptitle("Figure 4 — Sensor Distributions by Operating State",fontsize=12,fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/figures/04_boxplots_class.png",dpi=150,bbox_inches="tight"); plt.close()
print("  -> Figure 4 saved")

# 6. Correlation
print("\n[6/7] Correlation heatmap...")
s20=[s for s in sensor_cols if s in df.columns][:20]
cm=df[s20].corr()
hp=[(cm.columns[i],cm.columns[j],round(cm.iloc[i,j],3)) for i in range(len(cm.columns)) for j in range(i+1,len(cm.columns)) if abs(cm.iloc[i,j])>0.90]
print(f"  High-corr pairs (|r|>0.90, first 20 sensors): {len(hp)}")
fig,ax=plt.subplots(figsize=(13,11))
mask=np.triu(np.ones_like(cm,dtype=bool))
sns.heatmap(cm,mask=mask,cmap="RdBu_r",center=0,vmin=-1,vmax=1,square=True,linewidths=0.25,ax=ax,cbar_kws={"shrink":0.80,"label":"Pearson r"},annot=False)
ax.set_title("Figure 5 — Sensor-Sensor Correlation Matrix (sensor_00–sensor_19)",fontsize=12,fontweight="bold",pad=12)
plt.tight_layout(); plt.savefig("outputs/figures/05_correlation_heatmap.png",dpi=150,bbox_inches="tight"); plt.close()
print("  -> Figure 5 saved")

# 7. Summary
print("\n[7/7] Summary")
print("="*65)
print(f"  Rows          : {len(df):,}")
print(f"  Sensors       : {len(sensor_cols)}")
print(f"  Duration      : {duration} days")
for s in ["NORMAL","BROKEN","RECOVERING"]:
    print(f"  {s:<12}  : {cc[s]:,} ({cp[s]:.1f}%)")
print(f"  Imbalance     : {cc['NORMAL']/cc['BROKEN']:.1f}:1")
print(f"  Drop sensors  : {to_drop}")
print(f"  High-corr pairs: {len(hp)}")
print("="*65)
print("  All figures -> outputs/figures/")
print("  NEXT: notebooks/02_preprocessing.py")

summary={"n_rows":len(df),"n_sensors":len(sensor_cols),"duration_days":int(duration),
         "class_counts":cc.to_dict(),"class_pct":cp.round(2).to_dict(),
         "sensors_to_drop":to_drop,"good_sensors":gs,"high_corr_pairs":len(hp)}
os.makedirs("outputs",exist_ok=True)
with open("outputs/eda_summary.json","w") as f: json.dump(summary,f,indent=2)
print("  Summary JSON -> outputs/eda_summary.json")
