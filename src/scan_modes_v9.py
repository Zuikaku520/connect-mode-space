import nengo
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from collections import deque
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

# 基础单模式参数库
weights = np.array([0.2, 0.6, 1.0, 1.4, 1.8])

# 区域定义（Delayed + Recurrent）
REGIONS = {
    "HOTSPOT_A": {
        "delay_vals": np.array([0.03, 0.04, 0.05, 0.06, 0.07]),
        "gain_vals":  np.array([0.95, 0.97, 0.99, 0.995, 0.999]),
    },
    "HOTSPOT_B": {
        "delay_vals": np.array([0.09, 0.10, 0.12, 0.14, 0.16]),
        "gain_vals":  np.array([0.95, 0.97, 0.99, 0.995, 0.999]),
    },
    "CONTROL_C1": {
        "delay_vals": np.array([0.005, 0.008, 0.01, 0.012, 0.015]),
        "gain_vals":  np.array([0.75, 0.80, 0.85, 0.90, 0.94]),
    },
    "CONTROL_C2": {
        "delay_vals": np.array([0.30, 0.32, 0.35, 0.38, 0.40]),
        "gain_vals":  np.array([0.50, 0.55, 0.60, 0.65, 0.70]),
    },
    "CONTROL_C3": {
        "delay_vals": np.array([0.70, 0.75, 0.80, 0.85, 0.90]),
        "gain_vals":  np.array([0.10, 0.15, 0.20, 0.25, 0.30]),
    },
}


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
        recovery_slope = np.polyfit(x, y, 1)[0]
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

    return extract_metrics(sim.trange(), sim.data[probe])


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
    return summarize_metric_list(metric_list)


# ================== 建基础库 ==================
print("=" * 120)
print("BUILDING BASE LIBRARY FOR V9")
print("=" * 120)

results = {'WeightOnly': {}, 'Delayed': {}, 'Recurrent': {}, 'Delayed+Recurrent': {}}

all_delay_vals = sorted(set(np.concatenate([cfg["delay_vals"] for cfg in REGIONS.values()])))
all_gain_vals = sorted(set(np.concatenate([cfg["gain_vals"] for cfg in REGIONS.values()])))

for w in weights:
    mean_vec, std_vec = repeat_mode('WeightOnly', float(w))
    if mean_vec is not None:
        results['WeightOnly'][float(w)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"WeightOnly {w} done")

for d in all_delay_vals:
    mean_vec, std_vec = repeat_mode('Delayed', float(d))
    if mean_vec is not None:
        results['Delayed'][float(d)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"Delayed {d} done")

for g in all_gain_vals:
    mean_vec, std_vec = repeat_mode('Recurrent', float(g))
    if mean_vec is not None:
        results['Recurrent'][float(g)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"Recurrent {g} done")

for region_name, cfg in REGIONS.items():
    print("\n" + "=" * 120)
    print(f"SCANNING REGION: {region_name}")
    print("=" * 120)
    for d in cfg["delay_vals"]:
        for g in cfg["gain_vals"]:
            mean_vec, std_vec = repeat_mode('Delayed+Recurrent', (float(d), float(g)))
            if mean_vec is not None:
                results['Delayed+Recurrent'][(float(d), float(g))] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"row d={d} done")


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

def scan_region_to_maps(delay_vals, gain_vals):
    dist_map = np.full((len(delay_vals), len(gain_vals)), np.nan)
    syn_map = np.full((len(delay_vals), len(gain_vals)), np.nan)

    for i, d in enumerate(delay_vals):
        for j, g in enumerate(gain_vals):
            key = (float(d), float(g))
            if key not in results['Delayed+Recurrent']:
                continue

            combo = results['Delayed+Recurrent'][key]['scaled_mean_vec']
            delayed = results['Delayed'][float(d)]['scaled_mean_vec']
            recurrent = results['Recurrent'][float(g)]['scaled_mean_vec']

            dist_map[i, j] = nearest_weight_distance(combo)[1]
            syn_map[i, j] = synergy_index(combo, [delayed, recurrent])

    return dist_map, syn_map

def region_stats(name, dist_map, syn_map):
    valid_dist = dist_map[~np.isnan(dist_map)]
    valid_syn = syn_map[~np.isnan(syn_map)]

    print("\n" + "=" * 120)
    print(f"REGION QUANT SUMMARY: {name}")
    print("=" * 120)
    print(f"valid point count  : {len(valid_dist)}")
    print(f"synergy > 0 count  : {np.sum(valid_syn > 0)}")
    print(f"synergy > 1 count  : {np.sum(valid_syn > 1)}")
    print(f"synergy > 2 count  : {np.sum(valid_syn > 2)}")
    print(f"max synergy        : {np.max(valid_syn):.6f}")
    print(f"mean synergy       : {np.mean(valid_syn):.6f}")
    print(f"max dist           : {np.max(valid_dist):.6f}")
    print(f"mean dist          : {np.mean(valid_dist):.6f}")

def top_peaks(name, delay_vals, gain_vals, dist_map, syn_map, k=10):
    flat = []
    for i, d in enumerate(delay_vals):
        for j, g in enumerate(gain_vals):
            if np.isnan(dist_map[i, j]) or np.isnan(syn_map[i, j]):
                continue
            flat.append((d, g, syn_map[i, j], dist_map[i, j]))

    flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 120)
    print(f"TOP PEAKS: {name}")
    print("=" * 120)
    print(f"{'delay':>12} {'gain':>12} {'synergy':>12} {'dist_to_weight':>16}")
    print("-" * 120)
    for d, g, synv, distv in flat_sorted[:k]:
        print(f"{d:12.4f} {g:12.4f} {synv:12.6f} {distv:16.6f}")

def connected_components(mask):
    """4-neighborhood connected components on boolean mask"""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []

    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue
            q = deque([(i, j)])
            visited[i, j] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps

def print_component_stats(name, syn_map, threshold):
    valid_mask = ~np.isnan(syn_map)
    mask = valid_mask & (syn_map > threshold)
    comps = connected_components(mask)

    sizes = sorted([len(c) for c in comps], reverse=True)

    print("\n" + "=" * 120)
    print(f"CONNECTED COMPONENTS: {name}  (threshold = synergy > {threshold})")
    print("=" * 120)
    print(f"component count     : {len(comps)}")
    print(f"component sizes     : {sizes if sizes else []}")

def edge_profile(name, delay_vals, gain_vals, syn_map):
    """Simple edge profile: row and column means"""
    row_means = []
    for i, d in enumerate(delay_vals):
        vals = syn_map[i, :]
        vals = vals[~np.isnan(vals)]
        row_means.append(np.mean(vals) if len(vals) else np.nan)

    col_means = []
    for j, g in enumerate(gain_vals):
        vals = syn_map[:, j]
        vals = vals[~np.isnan(vals)]
        col_means.append(np.mean(vals) if len(vals) else np.nan)

    print("\n" + "=" * 120)
    print(f"EDGE PROFILE: {name}")
    print("=" * 120)
    print("Row mean synergy by delay:")
    for d, m in zip(delay_vals, row_means):
        print(f"  delay={d:.4f} -> mean_synergy={m:.6f}" if not np.isnan(m) else f"  delay={d:.4f} -> nan")

    print("Column mean synergy by gain:")
    for g, m in zip(gain_vals, col_means):
        print(f"  gain ={g:.4f} -> mean_synergy={m:.6f}" if not np.isnan(m) else f"  gain ={g:.4f} -> nan")


# ================== 输出各区域 ==================
for region_name, cfg in REGIONS.items():
    dist_map, syn_map = scan_region_to_maps(cfg["delay_vals"], cfg["gain_vals"])

    print_matrix(f"{region_name} : DISTANCE TO WEIGHTONLY", cfg["delay_vals"], cfg["gain_vals"], dist_map)
    print_matrix(f"{region_name} : SYNERGY INDEX", cfg["delay_vals"], cfg["gain_vals"], syn_map)

    region_stats(region_name, dist_map, syn_map)
    top_peaks(region_name, cfg["delay_vals"], cfg["gain_vals"], dist_map, syn_map, k=10)
    print_component_stats(region_name, syn_map, threshold=0.0)
    print_component_stats(region_name, syn_map, threshold=1.0)
    edge_profile(region_name, cfg["delay_vals"], cfg["gain_vals"], syn_map)

print("\nDone.")