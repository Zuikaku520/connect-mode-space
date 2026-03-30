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

N_REPEATS = 8          # 每个参数点重复次数
NOISE_STD = 0.02       # 输入噪声强度，可改成 0.01 / 0.03 测试
SEED_BASE = 42

# 单模式参数
weights = np.array([0.2, 0.6, 1.0, 1.4, 1.8])
delays = np.array([0.01, 0.05, 0.12, 0.35, 0.8])
gains = np.array([0.2, 0.6, 0.85, 0.94, 0.99])
inhibs = np.array([-0.2, -0.8, -1.4, -2.0, -2.6])

# 组合模式参数（控制计算量）
delayed_recurrent_grid = [
    (0.01, 0.85),
    (0.05, 0.6),
    (0.12, 0.85),
    (0.12, 0.99),
    (0.35, 0.2),
]

delayed_gated_grid = [
    (0.01, -0.2),
    (0.01, -2.0),
    (0.05, -0.2),
    (0.35, -1.4),
]

recurrent_gated_grid = [
    (0.2, -0.2),
    (0.2, -0.8),
    (0.6, -0.8),
    (0.85, -1.4),
    (0.99, -0.8),
]

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

    # steady state
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
print("RUNNING ROBUSTNESS SCAN")
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
            print(f"param={p} -> mean settling={mean_vec[5]:.6f}, mean auc={mean_vec[3]:.6f}, std auc={std_vec[3]:.6f}")


# ================== ROBUSTNESS SUMMARY ==================
print("\n" + "=" * 120)
print("ROBUSTNESS SUMMARY (mean ± std across repeats)")
print("=" * 120)

for mode_name, rows in results.items():
    print(f"\n{mode_name}")
    print("-" * 120)
    header = f"{'param':>28} " + " ".join([f"{k:>18}" for k in FEATURE_KEYS])
    print(header)
    print("-" * 120)

    for row in rows:
        p = row['param']
        mean_vec = row['mean_vec']
        std_vec = row['std_vec']
        vals = [f"{mean_vec[i]:.4f}±{std_vec[i]:.4f}" for i in range(len(FEATURE_KEYS))]
        print(f"{str(p):>28} " + " ".join([f"{v:>18}" for v in vals]))


# ================== 统一标准化后比较距离 ==================
all_mean_vecs = []
labels = []
params = []

for mode_name, rows in results.items():
    for row in rows:
        all_mean_vecs.append(row['mean_vec'])
        labels.append(mode_name)
        params.append(row['param'])

all_mean_vecs = np.array(all_mean_vecs, dtype=float)
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_mean_vecs)

scaled_rows = []
idx = 0
for mode_name, rows in results.items():
    for row in rows:
        row['scaled_mean_vec'] = all_scaled[idx]
        idx += 1


# ================== WeightOnly 替代测试 ==================
print("\n" + "=" * 120)
print("MODE REPLACEMENT TEST AGAINST WEIGHTONLY (ROBUST)")
print("=" * 120)

weight_rows = results['WeightOnly']

def nearest_weight_distance(target_vec):
    dists = []
    for wr in weight_rows:
        dists.append((wr['param'], euclidean(target_vec, wr['scaled_mean_vec'])))
    best = min(dists, key=lambda x: x[1])
    return best, dists

for mode_name, rows in results.items():
    if mode_name == 'WeightOnly':
        continue

    print(f"\n{mode_name} vs WeightOnly")
    print("-" * 120)
    print(f"{'target_param':>28} {'best_weight':>12} {'distance':>14}")
    print("-" * 120)

    dist_list = []
    for row in rows:
        best, _ = nearest_weight_distance(row['scaled_mean_vec'])
        dist_list.append(best[1])
        print(f"{str(row['param']):>28} {best[0]:12.4f} {best[1]:14.6f}")

    print("-" * 120)
    print(f"mean nearest distance = {np.mean(dist_list):.6f}")
    print(f"std  nearest distance = {np.std(dist_list):.6f}")
    print(f"max  nearest distance = {np.max(dist_list):.6f}")
    print(f"min  nearest distance = {np.min(dist_list):.6f}")


# ================== 消融测试 ==================
print("\n" + "=" * 120)
print("ABLATION TEST")
print("=" * 120)

def find_row(mode_name, target_param):
    for row in results[mode_name]:
        if row['param'] == target_param:
            return row
    return None

def ablation_compare(combo_mode, combo_param, base_candidates):
    combo_row = find_row(combo_mode, combo_param)
    if combo_row is None:
        print(f"Missing combo row: {combo_mode} {combo_param}")
        return

    print(f"\n{combo_mode}  param={combo_param}")
    print("-" * 120)
    print(f"{'compare_to':>28} {'param':>28} {'distance':>14}")

    for base_mode, base_param in base_candidates:
        base_row = find_row(base_mode, base_param)
        if base_row is None:
            print(f"{base_mode:>28} {str(base_param):>28} {'MISSING':>14}")
            continue

        d = euclidean(combo_row['scaled_mean_vec'], base_row['scaled_mean_vec'])
        print(f"{base_mode:>28} {str(base_param):>28} {d:14.6f}")

# Delayed+Recurrent
for p in delayed_recurrent_grid:
    d, g = p
    ablation_compare(
        'Delayed+Recurrent', p,
        [('Delayed', d), ('Recurrent', g)]
    )

# Delayed+Gated
for p in delayed_gated_grid:
    d, inh = p
    ablation_compare(
        'Delayed+Gated', p,
        [('Delayed', d), ('Gated', inh)]
    )

# Recurrent+Gated
for p in recurrent_gated_grid:
    g, inh = p
    ablation_compare(
        'Recurrent+Gated', p,
        [('Recurrent', g), ('Gated', inh)]
    )

# Triple ablation
for p in triple_grid:
    d, g, inh = p
    ablation_compare(
        'Delayed+Recurrent+Gated', p,
        [('Delayed', d), ('Recurrent', g), ('Gated', inh),
         ('Delayed+Recurrent', (d, g)),
         ('Delayed+Gated', (d, inh)),
         ('Recurrent+Gated', (g, inh))]
    )


# ================== 自动解释 ==================
print("\n" + "=" * 120)
print("AUTO INTERPRETATION")
print("=" * 120)

for mode_name, rows in results.items():
    if mode_name == 'WeightOnly':
        continue

    dist_list = []
    for row in rows:
        best, _ = nearest_weight_distance(row['scaled_mean_vec'])
        dist_list.append(best[1])

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