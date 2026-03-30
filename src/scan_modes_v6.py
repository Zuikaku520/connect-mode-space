import nengo
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ================== 参数设置 ==================
SIM_TIME = 2.5
DT = 0.001
PULSE_START = 0.1
PULSE_END = 0.11

N_REPEATS = 8
NOISE_STD = 0.02
SEED_BASE = 42

# 单模式参数
weights = np.array([0.2, 0.6, 1.0, 1.4, 1.8])
delays = np.array([0.01, 0.05, 0.12, 0.35, 0.8])
gains = np.array([0.2, 0.6, 0.85, 0.94, 0.99])
inhibs = np.array([-0.2, -0.8, -1.4, -2.0, -2.6])

# 二重组合：这次尽量补齐会在三重消融里用到的点
delayed_recurrent_grid = [
    (0.01, 0.85),
    (0.05, 0.6),
    (0.12, 0.85),
    (0.12, 0.99),
    (0.35, 0.2),
    (0.8, 0.99),      # 补齐
]

delayed_gated_grid = [
    (0.01, -0.2),
    (0.01, -1.4),     # 补齐
    (0.01, -2.0),
    (0.01, -2.6),     # 补齐
    (0.05, -0.2),
    (0.12, -0.2),     # 补齐
    (0.35, -1.4),
    (0.8, -0.2),      # 补齐
]

recurrent_gated_grid = [
    (0.2, -0.2),
    (0.2, -0.8),
    (0.6, -0.8),
    (0.85, -1.4),
    (0.85, -2.6),     # 补齐
    (0.94, -0.2),     # 补齐
    (0.99, -0.2),     # 补齐
    (0.99, -0.8),
    (0.99, -1.4),     # 补齐
]

# 三重组合
triple_grid = [
    (0.01, 0.85, -1.4),
    (0.01, 0.85, -2.6),
    (0.12, 0.99, -0.2),
    (0.8, 0.2, -0.2),
    (0.8, 0.99, -0.2),
]


# ================== 辅助函数 ==================
def ensure_1d(signal):
    signal = np.array(signal)
    if signal.ndim > 1:
        signal = signal.flatten()
    return signal


def make_pulse_generator(seed):
    rng = np.random.default_rng(seed)

    def pulse_generator(t):
        pulse = 1.0 if PULSE_START < t < PULSE_END else 0.0
        noise = rng.normal(0.0, NOISE_STD)
        return pulse + noise
    return pulse_generator


def extract_metrics(time, signal):
    signal = ensure_1d(signal)
    time = ensure_1d(time)

    mask = time > PULSE_START
    t_after = time[mask]
    s_after = signal[mask]

    if len(s_after) == 0:
        return None

    peak_idx = np.argmax(s_after)
    peak_time = t_after[peak_idx]
    peak_value = s_after[peak_idx]

    if peak_value < 1e-4:
        return {
            'peak_time': np.nan,
            'half_life': np.nan,
            'undershoot': np.nan,
            'auc': np.nan,
            'recovery_slope': np.nan,
            'settling_time': np.nan,
            'rise_time': np.nan,
            'energy': np.nan
        }

    # half-life
    half_val = peak_value / 2.0
    after_peak = s_after[peak_idx:]
    t_after_peak = t_after[peak_idx:]
    idx_half = np.where(after_peak <= half_val)[0]
    half_life = np.nan if len(idx_half) == 0 else (t_after_peak[idx_half[0]] - peak_time)

    # undershoot
    undershoot = np.min(signal) if np.min(signal) < 0 else 0.0

    # auc
    auc = np.trapz(np.abs(s_after), t_after)

    # recovery slope
    recovery_window = (t_after >= peak_time) & (t_after <= min(peak_time + 0.05, SIM_TIME))
    if np.sum(recovery_window) >= 3:
        x = t_after[recovery_window]
        y = s_after[recovery_window]
        coef = np.polyfit(x, y, 1)
        recovery_slope = coef[0]
    else:
        recovery_slope = np.nan

    # steady-state
    steady_mask = time > (SIM_TIME - 0.3)
    steady_val = np.mean(signal[steady_mask])

    # settling time
    tol = max(0.05 * np.abs(peak_value), 1e-4)
    settling_time = np.nan
    for i in range(peak_idx, len(s_after)):
        if np.all(np.abs(s_after[i:] - steady_val) < tol):
            settling_time = t_after[i] - peak_time
            break

    # rise time
    val10 = 0.1 * peak_value
    val90 = 0.9 * peak_value
    idx10 = np.where(s_after >= val10)[0]
    idx90 = np.where(s_after >= val90)[0]
    if len(idx10) > 0 and len(idx90) > 0:
        rise_time = t_after[idx90[0]] - t_after[idx10[0]]
    else:
        rise_time = np.nan

    # energy
    energy = np.trapz(s_after ** 2, t_after)

    return {
        'peak_time': peak_time,
        'half_life': half_life,
        'undershoot': undershoot,
        'auc': auc,
        'recovery_slope': recovery_slope,
        'settling_time': settling_time,
        'rise_time': rise_time,
        'energy': energy
    }


FEATURE_KEYS = [
    'peak_time', 'half_life', 'undershoot', 'auc',
    'recovery_slope', 'settling_time', 'rise_time', 'energy'
]


def metric_vector(m):
    return np.array([m[k] for k in FEATURE_KEYS], dtype=float)


def summarize_metric_list(metric_list):
    valid = []
    for m in metric_list:
        vec = metric_vector(m)
        if not np.any(np.isnan(vec)):
            valid.append(vec)

    if len(valid) == 0:
        return None, None

    valid = np.array(valid)
    return np.mean(valid, axis=0), np.std(valid, axis=0)


# ================== 仿真核心 ==================
def run_single_mode(mode_name, param, seed):
    with nengo.Network(seed=seed) as model:
        stimulus = nengo.Node(make_pulse_generator(seed))

        if mode_name == 'WeightOnly':
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, output, synapse=0.001, transform=float(param))
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Delayed':
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, output, synapse=float(param), transform=1.0)
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Recurrent':
            ens = nengo.Ensemble(100, dimensions=1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens, synapse=0.001, transform=1.0)
            nengo.Connection(ens, ens, synapse=0.05, transform=float(param))
            nengo.Connection(ens, output, synapse=0.001, transform=1.0)
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Gated':
            inhibitory = nengo.Ensemble(50, 1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, output, synapse=0.001, transform=0.5)
            nengo.Connection(stimulus, inhibitory, synapse=0.05, transform=1.0)
            nengo.Connection(inhibitory, output, synapse=0.01, transform=float(param))
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Delayed+Recurrent':
            d, g = param
            ens = nengo.Ensemble(100, dimensions=1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens, synapse=float(d), transform=1.0)
            nengo.Connection(ens, ens, synapse=0.05, transform=float(g))
            nengo.Connection(ens, output, synapse=0.001, transform=1.0)
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Delayed+Gated':
            d, inh = param
            inhibitory = nengo.Ensemble(50, 1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, output, synapse=float(d), transform=0.5)
            nengo.Connection(stimulus, inhibitory, synapse=0.05, transform=1.0)
            nengo.Connection(inhibitory, output, synapse=0.01, transform=float(inh))
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Recurrent+Gated':
            g, inh = param
            ens = nengo.Ensemble(100, dimensions=1)
            inhibitory = nengo.Ensemble(50, 1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens, synapse=0.001, transform=1.0)
            nengo.Connection(ens, ens, synapse=0.05, transform=float(g))
            nengo.Connection(ens, output, synapse=0.001, transform=0.7)
            nengo.Connection(stimulus, inhibitory, synapse=0.05, transform=1.0)
            nengo.Connection(inhibitory, output, synapse=0.01, transform=float(inh))
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Delayed+Recurrent+Gated':
            d, g, inh = param
            ens = nengo.Ensemble(100, dimensions=1)
            inhibitory = nengo.Ensemble(50, 1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens, synapse=float(d), transform=1.0)
            nengo.Connection(ens, ens, synapse=0.05, transform=float(g))
            nengo.Connection(ens, output, synapse=0.001, transform=0.7)
            nengo.Connection(stimulus, inhibitory, synapse=0.05, transform=1.0)
            nengo.Connection(inhibitory, output, synapse=0.01, transform=float(inh))
            probe = nengo.Probe(output, synapse=0.001)

        else:
            raise ValueError(f"Unknown mode: {mode_name}")

        with nengo.Simulator(model, dt=DT, progress_bar=False) as sim:
            sim.run(SIM_TIME)

    time = sim.trange()
    signal = sim.data[probe]
    return extract_metrics(time, signal)


def repeat_mode(mode_name, param, repeats=N_REPEATS):
    metric_list = []
    for i in range(repeats):
        seed = SEED_BASE + i
        try:
            m = run_single_mode(mode_name, param, seed)
            if m is not None:
                metric_list.append(m)
        except Exception as e:
            print(f"  ERROR in {mode_name} param={param} repeat={i}: {e}")
    mean_vec, std_vec = summarize_metric_list(metric_list)
    return metric_list, mean_vec, std_vec


# ================== 扫描 ==================
scan_dict = {
    'WeightOnly': list(weights),
    'Delayed': list(delays),
    'Recurrent': list(gains),
    'Gated': list(inhibs),
    'Delayed+Recurrent': delayed_recurrent_grid,
    'Delayed+Gated': delayed_gated_grid,
    'Recurrent+Gated': recurrent_gated_grid,
    'Delayed+Recurrent+Gated': triple_grid,
}

results = {}

print("=" * 120)
print("RUNNING V6 SCAN")
print("=" * 120)

for mode_name, param_list in scan_dict.items():
    print(f"\n### {mode_name}")
    results[mode_name] = []
    for p in param_list:
        metric_list, mean_vec, std_vec = repeat_mode(mode_name, p)
        if mean_vec is not None:
            results[mode_name].append({
                'param': p,
                'metric_list': metric_list,
                'mean_vec': mean_vec,
                'std_vec': std_vec
            })
            print(f"param={p} -> settling={mean_vec[5]:.6f}, auc={mean_vec[3]:.6f}, std_auc={std_vec[3]:.6f}")

# ================== 统一标准化 ==================
all_mean_vecs = []
for mode_name, rows in results.items():
    for row in rows:
        all_mean_vecs.append(row['mean_vec'])

all_mean_vecs = np.array(all_mean_vecs, dtype=float)
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_mean_vecs)

idx = 0
for mode_name, rows in results.items():
    for row in rows:
        row['scaled_mean_vec'] = all_scaled[idx]
        idx += 1


# ================== 查找函数 ==================
def find_row(mode_name, target_param):
    for row in results.get(mode_name, []):
        if row['param'] == target_param:
            return row
    return None


def nearest_weight_distance(target_vec):
    dists = []
    for wr in results['WeightOnly']:
        dists.append((wr['param'], euclidean(target_vec, wr['scaled_mean_vec'])))
    return min(dists, key=lambda x: x[1])


# ================== ROBUSTNESS SUMMARY ==================
print("\n" + "=" * 120)
print("ROBUSTNESS SUMMARY (mean ± std)")
print("=" * 120)

for mode_name, rows in results.items():
    print(f"\n{mode_name}")
    print("-" * 120)
    header = f"{'param':>28} " + " ".join([f"{k:>18}" for k in FEATURE_KEYS])
    print(header)
    print("-" * 120)
    for row in rows:
        vals = [f"{row['mean_vec'][i]:.4f}±{row['std_vec'][i]:.4f}" for i in range(len(FEATURE_KEYS))]
        print(f"{str(row['param']):>28} " + " ".join([f"{v:>18}" for v in vals]))


# ================== ROBUST WEIGHTONLY 替代测试 ==================
print("\n" + "=" * 120)
print("MODE REPLACEMENT TEST AGAINST WEIGHTONLY (V6)")
print("=" * 120)

mode_weight_distance_summary = {}

for mode_name, rows in results.items():
    if mode_name == 'WeightOnly':
        continue

    print(f"\n{mode_name} vs WeightOnly")
    print("-" * 120)
    print(f"{'target_param':>28} {'best_weight':>12} {'distance':>14} {'zone':>16}")
    print("-" * 120)

    dist_list = []
    for row in rows:
        best = nearest_weight_distance(row['scaled_mean_vec'])
        d = best[1]
        dist_list.append(d)

        if d < 1.2:
            zone = "near-baseline"
        elif d < 3.0:
            zone = "mid-shift"
        else:
            zone = "far-shift"

        print(f"{str(row['param']):>28} {best[0]:12.4f} {d:14.6f} {zone:>16}")

    mode_weight_distance_summary[mode_name] = dist_list

    print("-" * 120)
    print(f"mean nearest distance = {np.mean(dist_list):.6f}")
    print(f"std  nearest distance = {np.std(dist_list):.6f}")
    print(f"max  nearest distance = {np.max(dist_list):.6f}")
    print(f"min  nearest distance = {np.min(dist_list):.6f}")


# ================== 协同指数 ==================
print("\n" + "=" * 120)
print("SYNERGY INDEX TEST")
print("=" * 120)

def synergy_index(combo_row, base_rows):
    """
    synergy > 0: 组合比最接近的组成模式更远，说明更像新形态
    synergy < 0: 组合没有比组成模式更远
    """
    combo_to_weight = nearest_weight_distance(combo_row['scaled_mean_vec'])[1]
    base_to_weight = [nearest_weight_distance(br['scaled_mean_vec'])[1] for br in base_rows if br is not None]
    if len(base_to_weight) == 0:
        return np.nan
    return combo_to_weight - max(base_to_weight)

def print_synergy(combo_mode, combo_param, base_specs):
    combo_row = find_row(combo_mode, combo_param)
    if combo_row is None:
        return

    base_rows = []
    for base_mode, base_param in base_specs:
        base_rows.append(find_row(base_mode, base_param))

    s = synergy_index(combo_row, base_rows)

    print(f"{combo_mode:30s} param={str(combo_param):30s} synergy_index={s:.6f}" if not np.isnan(s)
          else f"{combo_mode:30s} param={str(combo_param):30s} synergy_index=NaN")

for p in delayed_recurrent_grid:
    d, g = p
    print_synergy('Delayed+Recurrent', p, [('Delayed', d), ('Recurrent', g)])

for p in delayed_gated_grid:
    d, inh = p
    print_synergy('Delayed+Gated', p, [('Delayed', d), ('Gated', inh)])

for p in recurrent_gated_grid:
    g, inh = p
    print_synergy('Recurrent+Gated', p, [('Recurrent', g), ('Gated', inh)])

for p in triple_grid:
    d, g, inh = p
    print_synergy(
        'Delayed+Recurrent+Gated',
        p,
        [('Delayed', d), ('Recurrent', g), ('Gated', inh)]
    )


# ================== 闭环消融测试 ==================
print("\n" + "=" * 120)
print("CLOSED-LOOP ABLATION TEST")
print("=" * 120)

def compare_dist(a_mode, a_param, b_mode, b_param):
    a = find_row(a_mode, a_param)
    b = find_row(b_mode, b_param)
    if a is None or b is None:
        return None
    return euclidean(a['scaled_mean_vec'], b['scaled_mean_vec'])

def closed_loop_ablation_triple(p):
    d, g, inh = p
    triple_mode = 'Delayed+Recurrent+Gated'

    candidates = [
        ('Delayed', d),
        ('Recurrent', g),
        ('Gated', inh),
        ('Delayed+Recurrent', (d, g)),
        ('Delayed+Gated', (d, inh)),
        ('Recurrent+Gated', (g, inh)),
    ]

    print(f"\n{triple_mode} param={p}")
    print("-" * 120)
    print(f"{'compare_to':>28} {'param':>28} {'distance':>14}")
    print("-" * 120)

    vals = []
    for cmode, cparam in candidates:
        dist = compare_dist(triple_mode, p, cmode, cparam)
        if dist is None:
            print(f"{cmode:>28} {str(cparam):>28} {'MISSING':>14}")
        else:
            vals.append((cmode, cparam, dist))
            print(f"{cmode:>28} {str(cparam):>28} {dist:14.6f}")

    if len(vals) > 0:
        nearest = min(vals, key=lambda x: x[2])
        farthest = max(vals, key=lambda x: x[2])
        print("-" * 120)
        print(f"nearest ablation target : {nearest[0]} {nearest[1]} dist={nearest[2]:.6f}")
        print(f"farthest ablation target: {farthest[0]} {farthest[1]} dist={farthest[2]:.6f}")

for p in triple_grid:
    closed_loop_ablation_triple(p)


# ================== 增量贡献分析 ==================
print("\n" + "=" * 120)
print("INCREMENTAL CONTRIBUTION TEST")
print("=" * 120)

def incremental_contribution_triple(p):
    d, g, inh = p
    triple_row = find_row('Delayed+Recurrent+Gated', p)
    if triple_row is None:
        return

    dr = find_row('Delayed+Recurrent', (d, g))
    dg = find_row('Delayed+Gated', (d, inh))
    rg = find_row('Recurrent+Gated', (g, inh))

    print(f"\nTriple param = {p}")
    print("-" * 120)
    print(f"{'base_combo':>22} {'distance_to_triple':>20}")

    distances = []
    for name, row in [('Delayed+Recurrent', dr), ('Delayed+Gated', dg), ('Recurrent+Gated', rg)]:
        if row is None:
            print(f"{name:>22} {'MISSING':>20}")
        else:
            dval = euclidean(triple_row['scaled_mean_vec'], row['scaled_mean_vec'])
            distances.append((name, dval))
            print(f"{name:>22} {dval:20.6f}")

    if len(distances) > 0:
        best = min(distances, key=lambda x: x[1])
        worst = max(distances, key=lambda x: x[1])
        print("-" * 120)
        print(f"closest lower-order combo : {best[0]}  dist={best[1]:.6f}")
        print(f"farthest lower-order combo: {worst[0]} dist={worst[1]:.6f}")

for p in triple_grid:
    incremental_contribution_triple(p)


# ================== AUTO INTERPRETATION ==================
print("\n" + "=" * 120)
print("AUTO INTERPRETATION")
print("=" * 120)

for mode_name, dist_list in mode_weight_distance_summary.items():
    mean_dist = np.mean(dist_list)

    if mean_dist > 4.5:
        verdict = "very strongly NOT replaceable by weight-only"
    elif mean_dist > 2.5:
        verdict = "strongly NOT replaceable by weight-only"
    elif mean_dist > 1.2:
        verdict = "partially not replaceable by weight-only"
    else:
        verdict = "may be partly approximated by weight-only"

    print(f"{mode_name:30s}: mean distance = {mean_dist:.6f} -> {verdict}")

print("\nDone.")