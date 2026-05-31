"""
3_KarlWeierstrass_NEXT5_2b3a — POGODNI deo iz 1_KarlWeierstrass_v2.py
Aparat 2b: Hurst / R-S analiza  +  Test 3a: Hurst eksponent

Self-contained:
  - KORAK 1: ucitavanje 4624 izvlacenja i izgradnja f(t) = lex-indeks
  - KORAK 2b: globalni Hurst nad f(t) (priprema)
  - KORAK 2b3a: rolling/local Hurst kroz vreme, shuffled Hurst referenca

Output:
  3_KarlWeierstrass_NEXT5_2b3a.png
  3_KarlWeierstrass_NEXT5_2b3a.txt
"""

import csv
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np


T0 = time.time()

CSV_DRAWS = "/data/loto7_4624_k43.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT5_2b3a.png")
TXT_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT5_2b3a.txt")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)


# ─── helperi (samo oni potrebni za 2b + 2b3a) ────────────────────────
def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if len(nums) == K_PICK and len(set(nums)) == K_PICK:
                rows.append(nums)
    return rows


def lex_rank_1based(combo, n=N_MAX, k=K_PICK):
    """1-based lex indeks (poklapa se sa rednim brojem u kombinacije_39C7.csv)."""
    combo = tuple(sorted(combo))
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo):
        remaining = k - i - 1
        for candidate in range(prev + 1, value):
            rank0 += math.comb(n - candidate, remaining)
        prev = value
    return rank0 + 1


def hurst_rs(series, min_window=8, max_window=None):
    """R/S Hurst procena: slope log(R/S) prema log(window)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if max_window is None:
        max_window = max(min_window * 2, n // 4)

    windows = []
    w = min_window
    while w <= max_window:
        windows.append(w)
        w = int(w * 1.45) + 1

    used_windows = []
    rs_values = []
    for w in windows:
        chunks = n // w
        if chunks < 2:
            continue
        vals = []
        for i in range(chunks):
            seg = x[i * w:(i + 1) * w]
            y = seg - seg.mean()
            z = np.cumsum(y)
            r = z.max() - z.min()
            s = seg.std(ddof=1)
            if s > 0:
                vals.append(r / s)
        if vals:
            used_windows.append(w)
            rs_values.append(float(np.mean(vals)))

    used_windows = np.asarray(used_windows, dtype=float)
    rs_values = np.asarray(rs_values, dtype=float)
    slope, intercept = np.polyfit(np.log(used_windows), np.log(rs_values), 1)
    fit = intercept + slope * np.log(used_windows)
    ss_res = float(np.sum((np.log(rs_values) - fit) ** 2))
    ss_tot = float(np.sum((np.log(rs_values) - np.log(rs_values).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2), used_windows, rs_values


def rolling_hurst_rs(series, window=768, step=128):
    """Rolling R/S Hurst procena kroz vreme."""
    x = np.asarray(series, dtype=float)
    centers = []
    hvals = []
    r2vals = []
    for start in range(0, len(x) - window + 1, step):
        seg = x[start:start + window]
        h, _, r2, _, _ = hurst_rs(seg, min_window=8, max_window=max(32, window // 4))
        centers.append(start + window // 2 + 1)
        hvals.append(h)
        r2vals.append(r2)
    return (
        np.asarray(centers, dtype=float),
        np.asarray(hvals, dtype=float),
        np.asarray(r2vals, dtype=float),
    )


# ─── KORAK 1: f(t) = lex-indeks ──────────────────────────────────────
draws = read_loto_csv(CSV_DRAWS)
N = len(draws)
lex_idx = np.array([lex_rank_1based(c) for c in draws], dtype=np.float64)

print()
print("3_KarlWeierstrass_NEXT5_2b3a — KORAK 1: formiranje krive f(t)")
print(f"  CSV:                  {CSV_DRAWS}")
print(f"  Ucitano izvlacenja:    {N}")
print(f"  C(39,7):              {TOTAL_COMBOS:,}")
print()

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("3_KarlWeierstrass_NEXT5_2b3a — Hurst/R-S + Hurst (POGODNO)\n")
    f.write("=" * 60 + "\n\n")
    f.write("KORAK 1: Weierstrass-ova funkcija nad svih izvucenih kombinacija\n\n")
    f.write(f"  CSV izvucenih:        {CSV_DRAWS}\n")
    f.write(f"  Ucitano izvlacenja:    {N}\n")
    f.write(f"  C(39,7):              {TOTAL_COMBOS:,}\n")
    f.write("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7\n\n")


# ─── KORAK 2b: globalni Hurst nad f(t) (priprema) ────────────────────
hurst_f, hurst_intercept, hurst_r2, hurst_windows, hurst_rs_values = hurst_rs(lex_idx)


# ─── KORAK 2b3a: rolling/local Hurst + shuffled referenca ────────────
T0_2B3A = time.time()

rolling_window = 768
rolling_step = 128
roll_centers, roll_h, roll_r2 = rolling_hurst_rs(
    lex_idx, window=rolling_window, step=rolling_step
)

rng_2b3a = np.random.default_rng(46)
hurst_shuffle_runs = 100
shuffle_h_f = []
for _ in range(hurst_shuffle_runs):
    shuffled_f = rng_2b3a.permutation(lex_idx)
    h_shuf, _, _, _, _ = hurst_rs(shuffled_f)
    shuffle_h_f.append(h_shuf)
shuffle_h_f = np.asarray(shuffle_h_f, dtype=float)

shuffle_h_mean = float(shuffle_h_f.mean())
shuffle_h_std = float(shuffle_h_f.std(ddof=1))
shuffle_h_p_high = float(np.mean(shuffle_h_f >= hurst_f))
shuffle_h_p_low = float(np.mean(shuffle_h_f <= hurst_f))
shuffle_h_z = (hurst_f - shuffle_h_mean) / (shuffle_h_std + 1e-12)

roll_h_mean = float(roll_h.mean())
roll_h_std = float(roll_h.std(ddof=1))
roll_h_min = float(roll_h.min())
roll_h_max = float(roll_h.max())
roll_persistent_count = int(np.sum(roll_h > 0.55))
roll_antipersistent_count = int(np.sum(roll_h < 0.45))

if shuffle_h_p_high <= 0.05 and hurst_f > shuffle_h_mean:
    hurst_2b3a_note = "globalni H je iznad shuffled reference"
elif shuffle_h_p_low <= 0.05 and hurst_f < shuffle_h_mean:
    hurst_2b3a_note = "globalni H je ispod shuffled reference"
else:
    hurst_2b3a_note = "globalni H nije jak odmak od shuffled reference"

print()
print("KORAK 2b3a: Aparat 2b Hurst/R-S + Test 3a Hurst eksponent")
print(f"  global H(f(t)) = {hurst_f:.4f}   R²={hurst_r2:.4f}")
print(f"  rolling H: mean={roll_h_mean:.4f} std={roll_h_std:.4f} "
      f"min={roll_h_min:.4f} max={roll_h_max:.4f}")
print(f"  rolling prozori: persistent={roll_persistent_count}/{len(roll_h)}  "
      f"anti={roll_antipersistent_count}/{len(roll_h)}")
print(f"  shuffled H: mean={shuffle_h_mean:.4f} std={shuffle_h_std:.4f} "
      f"z={shuffle_h_z:.2f} p_high={shuffle_h_p_high:.4f}")
print(f"  ⇒ {hurst_2b3a_note}")
print()

fig2b3a, ax2b3a = plt.subplots(1, 3, figsize=(16, 5))
fig2b3a.suptitle("KORAK 2b3a: Hurst/R-S aparat + Hurst test  (POGODNO)",
                 fontsize=13, fontweight="bold")

ax2b3a[0].plot(roll_centers, roll_h, "o-", markersize=3, color="darkslateblue")
ax2b3a[0].axhline(0.5, color="black", linestyle="--", linewidth=1.1, label="H=0.5")
ax2b3a[0].axhline(0.55, color="seagreen", linestyle=":", linewidth=1.0)
ax2b3a[0].axhline(0.45, color="crimson", linestyle=":", linewidth=1.0)
ax2b3a[0].set_title(f"Rolling Hurst (window={rolling_window}, step={rolling_step})")
ax2b3a[0].set_xlabel("t centar prozora")
ax2b3a[0].set_ylabel("H")
ax2b3a[0].legend(fontsize=8)
ax2b3a[0].grid(True, alpha=0.25)

ax2b3a[1].hist(shuffle_h_f, bins=22, color="lightgray", edgecolor="white")
ax2b3a[1].axvline(hurst_f, color="crimson", linewidth=2,
                  label=f"observed H={hurst_f:.3f}")
ax2b3a[1].axvline(shuffle_h_mean, color="black", linestyle="--",
                  label=f"shuffle mean={shuffle_h_mean:.3f}")
ax2b3a[1].set_title("Shuffled Hurst referenca")
ax2b3a[1].set_xlabel("H shuffled f(t)")
ax2b3a[1].set_ylabel("broj")
ax2b3a[1].legend(fontsize=8)

ax2b3a[2].plot(roll_centers, roll_r2, "o-", markersize=3, color="steelblue")
ax2b3a[2].set_ylim(0, 1.05)
ax2b3a[2].set_title("Kvalitet rolling R/S fit-a")
ax2b3a[2].set_xlabel("t centar prozora")
ax2b3a[2].set_ylabel("R²")
ax2b3a[2].grid(True, alpha=0.25)

for a in ax2b3a:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig2b3a.tight_layout()
fig2b3a.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
plt.show()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("KORAK 2b3a: Aparat 2b Hurst/R-S + Test 3a Hurst eksponent\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  PNG:                  {PNG_PATH}\n\n")
    f.write("Globalni Hurst nad f(t):\n")
    f.write(f"  H(f(t))               = {hurst_f:.8f}\n")
    f.write(f"  R^2                   = {hurst_r2:.8f}\n\n")
    f.write("Rolling/local Hurst:\n")
    f.write(f"  window                = {rolling_window}\n")
    f.write(f"  step                  = {rolling_step}\n")
    f.write(f"  broj prozora          = {len(roll_h)}\n")
    f.write(f"  mean H                = {roll_h_mean:.8f}\n")
    f.write(f"  std H                 = {roll_h_std:.8f}\n")
    f.write(f"  min H                 = {roll_h_min:.8f}\n")
    f.write(f"  max H                 = {roll_h_max:.8f}\n")
    f.write(f"  persistent H>0.55     = {roll_persistent_count}/{len(roll_h)}\n")
    f.write(f"  anti H<0.45           = {roll_antipersistent_count}/{len(roll_h)}\n\n")
    f.write("Shuffled Hurst referenca:\n")
    f.write(f"  runs                  = {hurst_shuffle_runs}\n")
    f.write(f"  mean                  = {shuffle_h_mean:.8f}\n")
    f.write(f"  std                   = {shuffle_h_std:.8f}\n")
    f.write(f"  z                     = {shuffle_h_z:.8f}\n")
    f.write(f"  p_high                = {shuffle_h_p_high:.8f}\n")
    f.write(f"  p_low                 = {shuffle_h_p_low:.8f}\n")
    f.write(f"  interpret.            = {hurst_2b3a_note}\n\n")
    f.write("Rolling H tacke:\n")
    f.write(f"  {'center':<10}{'H':>16}{'R^2':>16}\n")
    for center, hval, r2val in zip(roll_centers.astype(int), roll_h, roll_r2):
        f.write(f"  {center:<10}{hval:>16,.8f}{r2val:>16,.8f}\n")
    f.write("\n")

    elapsed_2b3a = time.time() - T0_2B3A
    f.write(f"Vreme KORAKA 2b3a: {timedelta(seconds=int(elapsed_2b3a))} ({elapsed_2b3a:.1f} s)\n")
    f.write(f"Ukupno vreme:       {timedelta(seconds=int(time.time()-T0))} ({time.time()-T0:.1f} s)\n")

print(f"PNG saved → {PNG_PATH}")
print(f"TXT saved → {TXT_PATH}")
print(f"Vreme KORAKA 2b3a: {timedelta(seconds=int(time.time()-T0_2B3A))} "
      f"({time.time()-T0_2B3A:.1f} s)")
print(f"Ukupno vreme:      {timedelta(seconds=int(time.time()-T0))} "
      f"({time.time()-T0:.1f} s)")
print()
print("KRAJ 3_KarlWeierstrass_NEXT5_2b3a.")
print()
"""
3_KarlWeierstrass_NEXT5_2b3a — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2b3a: Aparat 2b Hurst/R-S + Test 3a Hurst eksponent
  global H(f(t)) = 0.5931   R²=0.9988
  rolling H: mean=0.5974 std=0.0252 min=0.5494 max=0.6466
  rolling prozori: persistent=30/31  anti=0/31
  shuffled H: mean=0.5642 std=0.0173 z=1.67 p_high=0.0300
  ⇒ globalni H je iznad shuffled reference

PNG saved → /3_KarlWeierstrass_NEXT5_2b3a.png
TXT saved → /3_KarlWeierstrass_NEXT5_2b3a.txt
Vreme KORAKA 2b3a: 0:00:24 (24.3 s)
Ukupno vreme:      0:00:24 (24.4 s)

KRAJ 3_KarlWeierstrass_NEXT5_2b3a.
"""



###############   PREDIKCIJA 5  ###############################

"""
NEXT5 (2b3a, Hurst global+rolling) — trend extrapolacija perzistentnog režima H>0.5.
"""


def lex_unrank_1based(rank, n=N_MAX, k=K_PICK):
    """Vracanje 1-based lex indeksa u Loto 7/39 kombinaciju."""
    rank0 = int(rank) - 1
    combo = []
    prev = 0
    for i in range(k):
        remaining = k - i - 1
        for candidate in range(prev + 1, n + 1):
            count = math.comb(n - candidate, remaining)
            if rank0 >= count:
                rank0 -= count
            else:
                combo.append(candidate)
                prev = candidate
                break
    return tuple(combo)


T0_PRED5 = time.time()

# H>0.5 znaci perzistentan rezim: lokalni drift se ne gasi potpuno,
# vec se skalira jacom/slabijom udaljenoscu od Brown reference H=0.5.
last_lex = float(lex_idx[-1])
last_h = float(roll_h[-1])
global_h = float(hurst_f)
last_r2 = float(roll_r2[-1])

local_window = rolling_window
local_y = np.asarray(lex_idx[-local_window:], dtype=float)
local_x = np.arange(len(local_y), dtype=float)
local_slope, local_intercept = np.polyfit(local_x, local_y, 1)
local_fit = local_intercept + local_slope * local_x
local_resid = local_y - local_fit
local_resid_std = float(local_resid.std(ddof=1))

recent_incr = np.diff(local_y)
recent_mean_incr = float(recent_incr.mean())
last_incr = float(np.diff(lex_idx)[-1])

persistence_strength = float(np.clip((last_h - 0.5) / 0.15, 0.0, 1.0))
global_strength = float(np.clip((global_h - 0.5) / 0.15, 0.0, 1.0))
combined_strength = float((persistence_strength + global_strength) / 2.0)

# Blend: lokalni trend je osnova, zadnji inkrement ulazi samo koliko je rezim perzistentan.
pred_incr = (1.0 - combined_strength) * recent_mean_incr + combined_strength * last_incr
pred_lex_float = last_lex + pred_incr
pred_lex = int(np.clip(round(pred_lex_float), 1, TOTAL_COMBOS))
pred_combo = lex_unrank_1based(pred_lex)

z_grid = [-1.28, -0.84, -0.43, 0.0, 0.43, 0.84, 1.28]
candidate_rows = []
seen_lex = set()
for z in z_grid:
    cand_lex = int(np.clip(round(pred_lex_float + z * local_resid_std), 1, TOTAL_COMBOS))
    if cand_lex in seen_lex:
        continue
    seen_lex.add(cand_lex)
    candidate_rows.append((z, cand_lex, lex_unrank_1based(cand_lex)))

print()
print("PREDIKCIJA 5 — NEXT5 / 2b3a / Hurst perzistentni rezim")
print(f"  H global               = {global_h:.8f}")
print(f"  H zadnji rolling       = {last_h:.8f}")
print(f"  R2 zadnji rolling      = {last_r2:.8f}")
print(f"  persistence strength   = {combined_strength:.6f}")
print(f"  lokalni slope          = {local_slope:,.2f}")
print(f"  recent mean dX         = {recent_mean_incr:,.2f}")
print(f"  zadnji dX              = {last_incr:,.2f}")
print(f"  pred. inkrement        = {pred_incr:,.2f}")
print(f"  pred. lex              = {pred_lex:,}")
print(f"  pred. kombinacija      = {pred_combo}")
print("  kandidati oko Hurst-trend prognoze:")
for z, cand_lex, combo in candidate_rows:
    print(f"    z={z:>5.2f}  lex={cand_lex:>10,}  combo={combo}")
print()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("PREDIKCIJA 5: NEXT5 / 2b3a / Hurst perzistentni rezim\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model:\n")
    f.write("  H(f)>0.5 i skoro svi rolling prozori H>0.55 pokazuju perzistentan rezim.\n")
    f.write("  Lokalni trend u zadnjem rolling prozoru se kombinuje sa zadnjim inkrementom.\n")
    f.write("  Sto je H dalje iznad 0.5, to zadnji inkrement ima vecu tezinu.\n\n")
    f.write("Parametri:\n")
    f.write(f"  H global               = {global_h:.8f}\n")
    f.write(f"  H zadnji rolling       = {last_h:.8f}\n")
    f.write(f"  R2 zadnji rolling      = {last_r2:.8f}\n")
    f.write(f"  persistence strength   = {combined_strength:.8f}\n")
    f.write(f"  local window           = {local_window}\n")
    f.write(f"  lokalni slope          = {local_slope:,.8f}\n")
    f.write(f"  lokalni resid std      = {local_resid_std:,.8f}\n")
    f.write(f"  recent mean dX         = {recent_mean_incr:,.8f}\n")
    f.write(f"  zadnji dX              = {last_incr:,.8f}\n")
    f.write(f"  zadnji lex             = {int(last_lex):,}\n")
    f.write(f"  pred. inkrement        = {pred_incr:,.8f}\n\n")
    f.write("Glavna prognoza:\n")
    f.write(f"  pred. lex float        = {pred_lex_float:,.8f}\n")
    f.write(f"  pred. lex              = {pred_lex:,}\n")
    f.write(f"  pred. kombinacija      = {pred_combo}\n\n")
    f.write("Kandidati oko Hurst-trend prognoze:\n")
    f.write(f"  {'z':>8}{'lex':>14}  kombinacija\n")
    for z, cand_lex, combo in candidate_rows:
        f.write(f"  {z:>8.2f}{cand_lex:>14,}  {combo}\n")
    f.write("\n")
    elapsed_pred5 = time.time() - T0_PRED5
    f.write(f"Vreme PREDIKCIJE 5: {timedelta(seconds=int(elapsed_pred5))} ({elapsed_pred5:.1f} s)\n")

print(f"TXT updated → {TXT_PATH}")
print(f"Vreme PREDIKCIJE 5: {timedelta(seconds=int(time.time()-T0_PRED5))} "
      f"({time.time()-T0_PRED5:.1f} s)")
print()


"""
Predikcija iz Hurst perzistentnog režima: lokalni trend/drift iz rolling prozora skaliran snagom H signala.

Poslednji rolling H, lokalni linearni drift u zadnjem Hurst prozoru i perzistentnost H>0.5 kao težinu za sledeći lex.

koristi globalni H i poslednji rolling H
računa lokalni trend u zadnjem rolling prozoru
kombinuje lokalni prosečni inkrement i zadnji inkrement prema jačini perzistentnosti H > 0.5
vraća lex u Loto 7/39 kombinaciju
dodaje kandidate oko prognoze preko lokalne rezidualne širine
upisuje u 3_KarlWeierstrass_NEXT5_2b3a.txt
"""



"""
3_KarlWeierstrass_NEXT5_2b3a — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2b3a: Aparat 2b Hurst/R-S + Test 3a Hurst eksponent
  global H(f(t)) = 0.5931   R²=0.9988
  rolling H: mean=0.5974 std=0.0252 min=0.5494 max=0.6466
  rolling prozori: persistent=30/31  anti=0/31
  shuffled H: mean=0.5642 std=0.0173 z=1.67 p_high=0.0300
  ⇒ globalni H je iznad shuffled reference

PNG saved → /3_KarlWeierstrass_NEXT5_2b3a.png
TXT saved → /3_KarlWeierstrass_NEXT5_2b3a.txt
Vreme KORAKA 2b3a: 0:00:06 (6.0 s)
Ukupno vreme:      0:00:06 (6.0 s)

KRAJ 3_KarlWeierstrass_NEXT5_2b3a.


PREDIKCIJA 5 — NEXT5 / 2b3a / Hurst perzistentni rezim
  H global               = 0.59306018
  H zadnji rolling       = 0.62382045
  R2 zadnji rolling      = 0.99842733
  persistence strength   = 0.722935
  lokalni slope          = -739.26
  recent mean dX         = -6,859.23
  zadnji dX              = -2,143,496.00
  pred. inkrement        = -1,551,509.64
  pred. lex              = 1
  pred. kombinacija      = (1, 2, 3, 4, 5, 6, 7)
  kandidati oko Hurst-trend prognoze:
    z=-1.28  lex=         1  combo=(1, 2, 3, 4, 5, 6, 7)
    z= 0.43  lex=   896,898  combo=(1, x, 6, y, 27, z, 38)
    z= 0.84  lex= 2,742,179  combo=(1, x, 23, y, 25, z, 38)
    z= 1.28  lex= 4,722,480  combo=(2, x, 13, y, 28, z, 38)

TXT updated → /3_KarlWeierstrass_NEXT5_2b3a.txt
Vreme PREDIKCIJE 5: 0:00:00 (0.0 s)
"""
