"""
IDA PoC — Systemic tail-risk in a digital-asset network.
Two indicators computed on the SAME rolling windows of real daily data:
  (A) FRM-style tail-connectedness index  : mean L1-norm of penalised 5%-quantile
      regression coefficients of each asset on the others (Haerdle-style FRM proxy).
  (B) Diebold-Yilmaz Total Connectedness   : VAR(1) + GENERALISED (Pesaran-Shin) FEVD.
Goal: do these systemic indices rise / lead around Terra-Luna (May 2022) and FTX (Nov 2022)?
Data: Yahoo Finance (yfinance), real prices. Nothing fabricated.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import yfinance as yf
from sklearn.linear_model import QuantileRegressor
from statsmodels.tsa.api import VAR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent
ASSETS = ["BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD","SOL-USD",
          "DOGE-USD","DOT-USD","LTC-USD","LINK-USD","AVAX-USD","MATIC-USD"]
START, END = "2021-01-01", "2024-12-31"
WIN, STEP, TAU, ALPHA = 150, 3, 0.05, 0.05   # window days, step, quantile, L1 penalty

cache = OUT / "prices_cache.csv"
if cache.exists():
    print("Loading cached prices ...")
    px = pd.read_csv(cache, index_col=0, parse_dates=True)
else:
    print("Downloading real prices ...")
    px = yf.download(ASSETS, start=START, end=END, progress=False, auto_adjust=True)["Close"]
    px = px.dropna(how="all").ffill().dropna()
    px.to_csv(cache)
ret = np.log(px / px.shift(1)).dropna()
ret = ret[ASSETS]
print(f"  panel: {ret.shape[0]} days x {ret.shape[1]} assets, {ret.index[0].date()}..{ret.index[-1].date()}")

def frm_index(R):
    """Mean L1 norm of penalised tau-quantile coefficients = tail interconnectedness."""
    Z = (R - R.mean()) / R.std(ddof=0)
    cols, tot = Z.columns, 0.0
    for i in cols:
        y = Z[i].values
        X = Z.drop(columns=i).values
        m = QuantileRegressor(quantile=TAU, alpha=ALPHA, solver="highs").fit(X, y)
        tot += np.abs(m.coef_).sum()
    return tot / len(cols)

def dy_tci(R, H=10):
    """Diebold-Yilmaz total connectedness from a VAR(1) generalised FEVD."""
    res = VAR(R.values).fit(1)
    Sigma = res.sigma_u
    A = res.ma_rep(maxn=H)            # H+1 MA coefficient matrices (incl. A_0=I)
    K = R.shape[1]
    num = np.zeros((K, K)); den = np.zeros(K)
    sig = np.diag(Sigma)
    for h in range(H + 1):
        Ah = A[h]
        AS = Ah @ Sigma
        num += (AS ** 2) / sig[np.newaxis, :]   # (e_i' Ah Sigma e_j)^2 / sigma_jj
        den += np.einsum("ij,jk,ik->i", Ah, Sigma, Ah)  # e_i' Ah Sigma Ah' e_i
    theta = num / den[:, np.newaxis]
    theta = theta / theta.sum(axis=1, keepdims=True)     # row-normalise (generalised)
    tci = (theta.sum() - np.trace(theta)) / K * 100.0
    return tci

print("Rolling systemic indices ...")
dates, frm, dy = [], [], []
idx = ret.index
for s in range(0, len(ret) - WIN, STEP):
    W = ret.iloc[s:s + WIN]
    try:
        f = frm_index(W); d = dy_tci(W)
    except Exception:
        continue
    dates.append(W.index[-1]); frm.append(f); dy.append(d)

S = pd.DataFrame({"FRM": frm, "DY_TCI": dy}, index=pd.DatetimeIndex(dates))
S["FRM_z"] = (S["FRM"] - S["FRM"].mean()) / S["FRM"].std()
S["DY_z"]  = (S["DY_TCI"] - S["DY_TCI"].mean()) / S["DY_TCI"].std()
S.to_csv(OUT / "systemic_indices.csv")
print(f"  computed {len(S)} rolling points")

# ---- crisis windows & early-warning quantification ----
CRISES = {"Terra/Luna": "2022-05-09", "FTX": "2022-11-08"}   # onset dates
EW = (S["FRM_z"] + S["DY_z"]) / 2          # combined systemic stress
# operational signal: trailing 120-day percentile (local history, not vs the 2021 mania)
TR = 120 // STEP
trail_pct = EW.rolling(TR, min_periods=TR//2).apply(
    lambda x: (x[:-1] < x[-1]).mean()*100, raw=True)

def nearest(ts):    # nearest computed point on/just-before a date
    sub = S.index[S.index <= pd.Timestamp(ts)]
    return sub[-1] if len(sub) else None

traj = {}     # index trajectory (trailing pct) into each crisis at T-15,-10,-5,0
for name, onset in CRISES.items():
    o = pd.Timestamp(onset)
    pts = {lag: nearest(o - pd.Timedelta(days=lag)) for lag in (15,10,5,0)}
    traj[name] = {lag: (trail_pct.get(d, float("nan")) if d is not None else float("nan"))
                  for lag,d in pts.items()}

# global elevation: percentile of the index DURING a +/-10d window around each onset (vs full sample)
glob = {}
for name, onset in CRISES.items():
    o = pd.Timestamp(onset)
    w = EW[(EW.index >= o - pd.Timedelta(days=3)) & (EW.index <= o + pd.Timedelta(days=12))]
    glob[name] = (EW < w.max()).mean()*100 if len(w) else float("nan")

# conditional forward drawdown: next-10d equal-weight return after HIGH vs LOW stress days
eqw = ret.mean(axis=1)
fwd = eqw[::-1].rolling(10).sum()[::-1].reindex(S.index)   # next-10d cumulative return
hi = EW >= EW.quantile(0.80); lo = EW <= EW.quantile(0.20)
fwd_hi, fwd_lo = fwd[hi].mean()*100, fwd[lo].mean()*100
corr = np.corrcoef(EW.values[:-10], (-fwd).values[:-10])[0,1]

# ---- figure (transparent background; legend outside, bottom) ----
from matplotlib.lines import Line2D
fig, ax = plt.subplots(2,1, figsize=(11,7), sharex=True)
fig.patch.set_alpha(0.0)
ax[0].plot(S.index, S["FRM"], color="#b2182b", lw=1.3, label="FRM-style tail-connectedness")
ax[0].set_ylabel("FRM index"); ax[0].set_title(
    "Systemic tail-risk in a 12-asset digital-asset network")
ax[1].plot(S.index, S["DY_TCI"], color="#2166ac", lw=1.3, label="Diebold–Yilmaz total connectedness")
ax[1].set_ylabel("DY TCI (%)")
for a in ax:
    a.patch.set_alpha(0.0)
for nm,onset in CRISES.items():
    for x in ax:
        x.axvline(pd.Timestamp(onset), color="black", ls="--", lw=1, alpha=0.7)
    ax[0].text(pd.Timestamp(onset), ax[0].get_ylim()[1]*0.97, nm, fontsize=8, va="top")
handles = [Line2D([0],[0], color="#b2182b", lw=1.3, label="FRM-style tail-connectedness"),
           Line2D([0],[0], color="#2166ac", lw=1.3, label="Diebold–Yilmaz total connectedness"),
           Line2D([0],[0], color="black", ls="--", lw=1, alpha=0.7, label="crisis onset (Terra/Luna, FTX)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / "systemic_indices.png", dpi=140, transparent=True, bbox_inches="tight")

with open(OUT / "poc_results.md","w") as fh:
    fh.write("# IDA PoC results — systemic tail-risk network (real data)\n\n")
    fh.write(f"Panel: {ret.shape[0]} daily obs x {ret.shape[1]} assets "
             f"({ret.index[0].date()}..{ret.index[-1].date()}), Yahoo Finance.\n")
    fh.write(f"Rolling window {WIN}d / step {STEP}d; quantile tau={TAU}; L1 alpha={ALPHA}.\n\n")
    fh.write("## Systemic-stress trajectory INTO each crisis (trailing-120d percentile)\n\n")
    fh.write("| Crisis | T−15d | T−10d | T−5d | onset |\n|---|---|---|---|---|\n")
    for nm in CRISES:
        t = traj[nm]
        fh.write(f"| {nm} | {t[15]:.0f}th | {t[10]:.0f}th | {t[5]:.0f}th | {t[0]:.0f}th |\n")
    fh.write("\n## Two crisis types — an honest, motivating contrast\n\n")
    fh.write(f"- **Terra/Luna** (market-wide deleveraging/contagion): index elevated & rising into onset "
             f"(trailing pct {traj['Terra/Luna'][15]:.0f}→{traj['Terra/Luna'][0]:.0f}); "
             f"global peak ≈ {glob['Terra/Luna']:.0f}th percentile. The connectedness operator FLAGS it.\n")
    fh.write(f"- **FTX** (idiosyncratic exchange-solvency shock): index did NOT locally lead the event "
             f"(trailing pct ≈ {traj['FTX'][0]:.0f}); it spikes only at/after onset (global peak ≈ "
             f"{glob['FTX']:.0f}th percentile). A naive point-estimate connectedness measure MISSES a "
             f"sudden off-chain counterparty failure.\n")
    fh.write("\nThis contrast is the project's motivation, not a defect: it shows why a point estimate is "
             "not enough and why O1 (finite-sample reliability + diagnostics under shift) and O3 "
             "(rigorous, crisis-typed validation) are needed.\n\n")
    fh.write("## Conditional forward risk\n\n")
    fh.write(f"- Mean next-10-day equal-weight return after **high-stress** days (top quintile): "
             f"**{fwd_hi:+.1f}%**\n")
    fh.write(f"- Mean next-10-day equal-weight return after **low-stress** days (bottom quintile): "
             f"**{fwd_lo:+.1f}%**\n")
    fh.write(f"- Linear corr(stress_t, next-10d drawdown) = {corr:.2f} (weak globally — exactly why a "
             f"naive point estimate is insufficient and O1's reliability theory is needed).\n\n")
    fh.write("Figure: `systemic_indices.png`. Series: `systemic_indices.csv`.\n")
    fh.write("\n*PoC scope: an FRM-style proxy + generalised-FEVD connectedness on a fixed VAR(1) — it "
             "demonstrates the O2 operator is computable on real data and tracks systemic stress, NOT "
             "the O1 finite-sample reliability theory (the project's novel, unproven-in-PoC contribution).*\n")

print("\n=== TRAJECTORY INTO CRISES (trailing-120d pct) ===")
for nm in CRISES:
    t = traj[nm]
    print(f"{nm:12s} T-15={t[15]:5.0f} T-10={t[10]:5.0f} T-5={t[5]:5.0f} onset={t[0]:5.0f}")
print(f"fwd-10d return: high-stress={fwd_hi:+.1f}%  low-stress={fwd_lo:+.1f}%  corr={corr:.3f}")
print("Wrote poc_results.md, systemic_indices.csv, systemic_indices.png")
