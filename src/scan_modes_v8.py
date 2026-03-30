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

N_REPEATS = 6
NOISE_STD = 0.02
SEED_BASE = 42

FEATURE_KEYS = [
    'peak_time', 'half_life', 'undershoot', 'auc',
    'recovery_slope', 'settling_time', 'rise_time', 'energy'
]

# 基础单模式库
weights = np.array([0.2, 0.6, 1.0, 1.4, 1.8])
delay_base = np.array([0.01, 0.05, 0.12, 0.35, 0.8])
gain_base = np.array([0.2, 0.6, 0.85, 0.94, 0.99])

# 局部精扫描区域
# 热点A附近
delay_fine_A = np.array([0.03, 0.04, 0.05, 0.06, 0.07])
gain_fine_A  = np.array([0.95, 0.97, 0.99, 0.995, 0.999])

# 热点B附近
delay_fine_B = np.array([0.09, 0.10, 0.12, 0.14, 0.16])
gain_fine_B  = np.array([0.95, 0.97, 0.99, 0.995, 0.999])

# 对照区
delay_fine_C = np.array([0.005, 0.008, 0.01, 0.012, 0.015])
gain_fine_C  = np.array([0.75, 0.80, 0.85, 0.90, 0.94])


# ================== 基础函数 ==================
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
        return {k: np.nan for k in FEATURE_KEYS}

    half_val = peak_value / 2.0
    after_peak = s_after[peak_idx:]
    t_after_peak = t_after[peak_idx:]
    idx_half = np.where(after_peak <= half_val)[0]
    half_life = np.nan if len(idx_half) == 0 else (t_after_peak[idx_half[0]] - peak_time)

    undershoot = np.min(signal) if np.min(signal) < 0 else 0.0
    auc = np.trapz(np.abs(s_after), t_after)

    recovery_window = (t_after >= peak_time) & (t_after <= min(peak_time + 0.05, SIM_TIME))
    if np.sum(recovery_window) >= 3:
        x = t_after[recovery_window]
        y = s_after[recovery_window]
        coef = np.polyfit(x, y, 1)
        recovery_slope = coef[0]
    else:
        recovery_slope = np.nan

    steady_mask = time > (SIM_TIME - 0.3)
    steady_val = np.mean(signal[steady_mask])

    tol = max(0.05 * np.abs(peak_value), 1e-4)
    settling_time = np.nan
    for i in range(peak_idx, len(s_after)):
        if np.all(np.abs(s_after[i:] - steady_val) < tol):
            settling_time = t_after[i] - peak_time
            break

    val10 = 0.1 * peak_value
    val90 = 0.9 * peak_value
    idx10 = np.where(s_after >= val10)[0]
    idx90 = np.where(s_after >= val90)[0]
    if len(idx10) > 0 and len(idx90) > 0:
        rise_time = t_after[idx90[0]] - t_after[idx10[0]]
    else:
        rise_time = np.nan

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


# ================== 仿真器 ==================
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

        elif mode_name == 'Delayed+Recurrent':
            d, g = param
            ens = nengo.Ensemble(100, dimensions=1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens, synapse=float(d), transform=1.0)
            nengo.Connection(ens, ens, synapse=0.05, transform=float(g))
            nengo.Connection(ens, output, synapse=0.001, transform=1.0)
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
            print(f"ERROR: {mode_name} {param} repeat={i}: {e}")
    mean_vec, std_vec = summarize_metric_list(metric_list)
    return mean_vec, std_vec


# ================== 建基础库 ==================
print("=" * 120)
print("BUILDING BASE LIBRARY FOR V8")
print("=" * 120)

results = {'WeightOnly': {}, 'Delayed': {}, 'Recurrent': {}, 'Delayed+Recurrent': {}}

for w in weights:
    mean_vec, std_vec = repeat_mode('WeightOnly', float(w))
    if mean_vec is not None:
        results['WeightOnly'][float(w)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"WeightOnly {w} done")

for d in np.unique(np.concatenate([delay_base, delay_fine_A, delay_fine_B, delay_fine_C])):
    mean_vec, std_vec = repeat_mode('Delayed', float(d))
    if mean_vec is not None:
        results['Delayed'][float(d)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"Delayed {d} done")

for g in np.unique(np.concatenate([gain_base, gain_fine_A, gain_fine_B, gain_fine_C])):
    mean_vec, std_vec = repeat_mode('Recurrent', float(g))
    if mean_vec is not None:
        results['Recurrent'][float(g)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"Recurrent {g} done")


def build_dr_region(delay_vals, gain_vals, region_name):
    print("\n" + "=" * 120)
    print(f"SCANNING REGION: {region_name}")
    print("=" * 120)

    region = {}
    for d in delay_vals:
        for g in gain_vals:
            mean_vec, std_vec = repeat_mode('Delayed+Recurrent', (float(d), float(g)))
            if mean_vec is not None:
                region[(float(d), float(g))] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"row d={d} done")
    return region


region_A = build_dr_region(delay_fine_A, gain_fine_A, "HOTSPOT_A around (0.05, 0.99)")
region_B = build_dr_region(delay_fine_B, gain_fine_B, "HOTSPOT_B around (0.12, 0.99)")
region_C = build_dr_region(delay_fine_C, gain_fine_C, "CONTROL_C around (0.01, 0.85)")

# 合并进 results
results['Delayed+Recurrent'].update(region_A)
results['Delayed+Recurrent'].update(region_B)
results['Delayed+Recurrent'].update(region_C)

# ================== 标准化 ==================
all_vecs = []
refs = []

for mode_name, table in results.items():
    for param, item in table.items():
        all_vecs.append(item['mean_vec'])
        refs.append((mode_name, param))

all_vecs = np.array(all_vecs, dtype=float)
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_vecs)

for idx, (mode_name, param) in enumerate(refs):
    results[mode_name][param]['scaled_mean_vec'] = all_scaled[idx]


# ================== 工具函数 ==================
def nearest_weight_distance(target_vec):
    dists = []
    for w, item in results['WeightOnly'].items():
        dists.append((w, euclidean(target_vec, item['scaled_mean_vec'])))
    return min(dists, key=lambda x: x[1])

def synergy_index(combo_vec, parent_vecs):
    combo_weight_dist = nearest_weight_distance(combo_vec)[1]
    parent_weight_dists = [nearest_weight_distance(v)[1] for v in parent_vecs]
    return combo_weight_dist - max(parent_weight_dists)

def print_matrix(title, row_values, col_values, matrix, fmt="{:8.3f}"):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    header = f"{'row\\col':>12}"
    for c in col_values:
        header += f"{str(c):>12}"
    print(header)
    print("-" * 120)

    for i, r in enumerate(row_values):
        line = f"{str(r):>12}"
        for j in range(len(col_values)):
            v = matrix[i, j]
            if isinstance(v, float) and np.isnan(v):
                line += f"{'nan':>12}"
            else:
                line += f"{fmt.format(v):>12}"
        print(line)

def scan_region_to_maps(region, delay_vals, gain_vals):
    dist_map = np.full((len(delay_vals), len(gain_vals)), np.nan)
    syn_map = np.full((len(delay_vals), len(gain_vals)), np.nan)

    for i, d in enumerate(delay_vals):
        for j, g in enumerate(gain_vals):
            key = (float(d), float(g))
            if key not in region:
                continue

            combo = region[key]['scaled_mean_vec']
            delayed = results['Delayed'][float(d)]['scaled_mean_vec']
            recurrent = results['Recurrent'][float(g)]['scaled_mean_vec']

            dist_map[i, j] = nearest_weight_distance(combo)[1]
            syn_map[i, j] = synergy_index(combo, [delayed, recurrent])

    return dist_map, syn_map

def top_k_from_region(name, delay_vals, gain_vals, dist_map, syn_map, k=10):
    flat = []
    for i, d in enumerate(delay_vals):
        for j, g in enumerate(gain_vals):
            if np.isnan(dist_map[i, j]) or np.isnan(syn_map[i, j]):
                continue
            flat.append((d, g, syn_map[i, j], dist_map[i, j]))

    flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 120)
    print(f"TOP {k} SYNERGY POINTS: {name}")
    print("=" * 120)
    print(f"{'delay':>12} {'gain':>12} {'synergy':>12} {'dist_to_weight':>16}")
    print("-" * 120)
    for d, g, synv, distv in flat_sorted[:k]:
        print(f"{d:12.4f} {g:12.4f} {synv:12.6f} {distv:16.6f}")

def region_stats(name, dist_map, syn_map):
    valid_dist = dist_map[~np.isnan(dist_map)]
    valid_syn = syn_map[~np.isnan(syn_map)]

    print("\n" + "=" * 120)
    print(f"REGION STATS: {name}")
    print("=" * 120)
    print(f"valid point count  : {len(valid_dist)}")
    print(f"synergy > 0 count  : {np.sum(valid_syn > 0)}")
    print(f"synergy > 1 count  : {np.sum(valid_syn > 1)}")
    print(f"synergy > 2 count  : {np.sum(valid_syn > 2)}")
    print(f"max synergy        : {np.max(valid_syn):.6f}")
    print(f"mean synergy       : {np.mean(valid_syn):.6f}")
    print(f"max dist           : {np.max(valid_dist):.6f}")
    print(f"mean dist          : {np.mean(valid_dist):.6f}")


# ================== 区域地图输出 ==================
dist_A, syn_A = scan_region_to_maps(region_A, delay_fine_A, gain_fine_A)
dist_B, syn_B = scan_region_to_maps(region_B, delay_fine_B, gain_fine_B)
dist_C, syn_C = scan_region_to_maps(region_C, delay_fine_C, gain_fine_C)

print_matrix("HOTSPOT_A : DISTANCE TO WEIGHTONLY", delay_fine_A, gain_fine_A, dist_A)
print_matrix("HOTSPOT_A : SYNERGY INDEX", delay_fine_A, gain_fine_A, syn_A)

print_matrix("HOTSPOT_B : DISTANCE TO WEIGHTONLY", delay_fine_B, gain_fine_B, dist_B)
print_matrix("HOTSPOT_B : SYNERGY INDEX", delay_fine_B, gain_fine_B, syn_B)

print_matrix("CONTROL_C : DISTANCE TO WEIGHTONLY", delay_fine_C, gain_fine_C, dist_C)
print_matrix("CONTROL_C : SYNERGY INDEX", delay_fine_C, gain_fine_C, syn_C)

top_k_from_region("HOTSPOT_A", delay_fine_A, gain_fine_A, dist_A, syn_A, k=10)
top_k_from_region("HOTSPOT_B", delay_fine_B, gain_fine_B, dist_B, syn_B, k=10)
top_k_from_region("CONTROL_C", delay_fine_C, gain_fine_C, dist_C, syn_C, k=10)

region_stats("HOTSPOT_A", dist_A, syn_A)
region_stats("HOTSPOT_B", dist_B, syn_B)
region_stats("CONTROL_C", dist_C, syn_C)

print("\nDone.")