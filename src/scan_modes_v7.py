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

# 单模式参数
weights = np.array([0.2, 0.6, 1.0, 1.4, 1.8])

delay_grid = np.array([0.01, 0.05, 0.12, 0.35, 0.8])
gain_grid = np.array([0.2, 0.6, 0.85, 0.94, 0.99])
inhib_grid = np.array([-0.2, -0.8, -1.4, -2.0, -2.6])


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


# ================== 先跑基础库 ==================
print("=" * 120)
print("BUILDING BASE LIBRARY")
print("=" * 120)

results = {
    'WeightOnly': {},
    'Delayed': {},
    'Recurrent': {},
    'Gated': {},
    'Delayed+Recurrent': {},
    'Delayed+Gated': {},
    'Recurrent+Gated': {},
}

# WeightOnly
for w in weights:
    mean_vec, std_vec = repeat_mode('WeightOnly', float(w))
    if mean_vec is not None:
        results['WeightOnly'][float(w)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"WeightOnly {w} done")

# 单模式
for d in delay_grid:
    mean_vec, std_vec = repeat_mode('Delayed', float(d))
    if mean_vec is not None:
        results['Delayed'][float(d)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"Delayed {d} done")

for g in gain_grid:
    mean_vec, std_vec = repeat_mode('Recurrent', float(g))
    if mean_vec is not None:
        results['Recurrent'][float(g)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"Recurrent {g} done")

for inh in inhib_grid:
    mean_vec, std_vec = repeat_mode('Gated', float(inh))
    if mean_vec is not None:
        results['Gated'][float(inh)] = {'mean_vec': mean_vec, 'std_vec': std_vec}
        print(f"Gated {inh} done")

# 二重模式全二维平面
for d in delay_grid:
    for g in gain_grid:
        mean_vec, std_vec = repeat_mode('Delayed+Recurrent', (float(d), float(g)))
        if mean_vec is not None:
            results['Delayed+Recurrent'][(float(d), float(g))] = {'mean_vec': mean_vec, 'std_vec': std_vec}
    print(f"Delayed+Recurrent row d={d} done")

for d in delay_grid:
    for inh in inhib_grid:
        mean_vec, std_vec = repeat_mode('Delayed+Gated', (float(d), float(inh)))
        if mean_vec is not None:
            results['Delayed+Gated'][(float(d), float(inh))] = {'mean_vec': mean_vec, 'std_vec': std_vec}
    print(f"Delayed+Gated row d={d} done")

for g in gain_grid:
    for inh in inhib_grid:
        mean_vec, std_vec = repeat_mode('Recurrent+Gated', (float(g), float(inh)))
        if mean_vec is not None:
            results['Recurrent+Gated'][(float(g), float(inh))] = {'mean_vec': mean_vec, 'std_vec': std_vec}
    print(f"Recurrent+Gated row g={g} done")


# ================== 标准化 ==================
all_vecs = []
vec_refs = []

for mode_name, table in results.items():
    for param, item in table.items():
        all_vecs.append(item['mean_vec'])
        vec_refs.append((mode_name, param))

all_vecs = np.array(all_vecs, dtype=float)
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_vecs)

for idx, (mode_name, param) in enumerate(vec_refs):
    results[mode_name][param]['scaled_mean_vec'] = all_scaled[idx]


# ================== 工具函数 ==================
def nearest_weight_distance(target_vec):
    dists = []
    for w, item in results['WeightOnly'].items():
        dists.append((w, euclidean(target_vec, item['scaled_mean_vec'])))
    return min(dists, key=lambda x: x[1])

def mode_zone(dist):
    if dist < 1.2:
        return "near-baseline"
    elif dist < 3.0:
        return "mid-shift"
    else:
        return "far-shift"

def synergy_index(combo_vec, parent_vecs):
    parent_weight_dists = [nearest_weight_distance(v)[1] for v in parent_vecs]
    combo_weight_dist = nearest_weight_distance(combo_vec)[1]
    return combo_weight_dist - max(parent_weight_dists)

def nearest_parent_name(combo_vec, parent_dict):
    pairs = []
    for name, vec in parent_dict.items():
        pairs.append((name, euclidean(combo_vec, vec)))
    return min(pairs, key=lambda x: x[1])

def safe_get_result(mode_name, param):
    return results.get(mode_name, {}).get(param, None)


# ================== 地图打印函数 ==================
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
        for j, _ in enumerate(col_values):
            v = matrix[i, j]
            if isinstance(v, str):
                line += f"{v:>12}"
            elif isinstance(v, float) and np.isnan(v):
                line += f"{'nan':>12}"
            else:
                line += f"{fmt.format(v):>12}"
        print(line)


# ================== 1) Delayed + Recurrent 协同地图 ==================
dr_dist_map = np.zeros((len(delay_grid), len(gain_grid)))
dr_syn_map = np.zeros((len(delay_grid), len(gain_grid)))
dr_parent_map = np.empty((len(delay_grid), len(gain_grid)), dtype=object)

for i, d in enumerate(delay_grid):
    for j, g in enumerate(gain_grid):
        combo_row = safe_get_result('Delayed+Recurrent', (float(d), float(g)))
        delayed_row = safe_get_result('Delayed', float(d))
        recurrent_row = safe_get_result('Recurrent', float(g))

        if combo_row is None or delayed_row is None or recurrent_row is None:
            dr_dist_map[i, j] = np.nan
            dr_syn_map[i, j] = np.nan
            dr_parent_map[i, j] = "MISSING"
            continue

        combo = combo_row['scaled_mean_vec']
        delayed = delayed_row['scaled_mean_vec']
        recurrent = recurrent_row['scaled_mean_vec']

        dr_dist_map[i, j] = nearest_weight_distance(combo)[1]
        dr_syn_map[i, j] = synergy_index(combo, [delayed, recurrent])
        dr_parent_map[i, j] = nearest_parent_name(combo, {
            'Delayed': delayed,
            'Recurrent': recurrent
        })[0]

print_matrix("DELAYED + RECURRENT : DISTANCE TO WEIGHTONLY", delay_grid, gain_grid, dr_dist_map)
print_matrix("DELAYED + RECURRENT : SYNERGY INDEX", delay_grid, gain_grid, dr_syn_map)
print_matrix("DELAYED + RECURRENT : NEAREST PARENT", delay_grid, gain_grid, dr_parent_map)


# ================== 2) Delayed + Gated 协同地图 ==================
dg_dist_map = np.zeros((len(delay_grid), len(inhib_grid)))
dg_syn_map = np.zeros((len(delay_grid), len(inhib_grid)))
dg_parent_map = np.empty((len(delay_grid), len(inhib_grid)), dtype=object)

for i, d in enumerate(delay_grid):
    for j, inh in enumerate(inhib_grid):
        combo_row = safe_get_result('Delayed+Gated', (float(d), float(inh)))
        delayed_row = safe_get_result('Delayed', float(d))
        gated_row = safe_get_result('Gated', float(inh))

        if combo_row is None or delayed_row is None or gated_row is None:
            dg_dist_map[i, j] = np.nan
            dg_syn_map[i, j] = np.nan
            dg_parent_map[i, j] = "MISSING"
            continue

        combo = combo_row['scaled_mean_vec']
        delayed = delayed_row['scaled_mean_vec']
        gated = gated_row['scaled_mean_vec']

        dg_dist_map[i, j] = nearest_weight_distance(combo)[1]
        dg_syn_map[i, j] = synergy_index(combo, [delayed, gated])
        dg_parent_map[i, j] = nearest_parent_name(combo, {
            'Delayed': delayed,
            'Gated': gated
        })[0]

print_matrix("DELAYED + GATED : DISTANCE TO WEIGHTONLY", delay_grid, inhib_grid, dg_dist_map)
print_matrix("DELAYED + GATED : SYNERGY INDEX", delay_grid, inhib_grid, dg_syn_map)
print_matrix("DELAYED + GATED : NEAREST PARENT", delay_grid, inhib_grid, dg_parent_map)


# ================== 3) Recurrent + Gated 协同地图 ==================
rg_dist_map = np.zeros((len(gain_grid), len(inhib_grid)))
rg_syn_map = np.zeros((len(gain_grid), len(inhib_grid)))
rg_parent_map = np.empty((len(gain_grid), len(inhib_grid)), dtype=object)

for i, g in enumerate(gain_grid):
    for j, inh in enumerate(inhib_grid):
        combo_row = safe_get_result('Recurrent+Gated', (float(g), float(inh)))
        recurrent_row = safe_get_result('Recurrent', float(g))
        gated_row = safe_get_result('Gated', float(inh))

        if combo_row is None or recurrent_row is None or gated_row is None:
            rg_dist_map[i, j] = np.nan
            rg_syn_map[i, j] = np.nan
            rg_parent_map[i, j] = "MISSING"
            continue

        combo = combo_row['scaled_mean_vec']
        recurrent = recurrent_row['scaled_mean_vec']
        gated = gated_row['scaled_mean_vec']

        rg_dist_map[i, j] = nearest_weight_distance(combo)[1]
        rg_syn_map[i, j] = synergy_index(combo, [recurrent, gated])
        rg_parent_map[i, j] = nearest_parent_name(combo, {
            'Recurrent': recurrent,
            'Gated': gated
        })[0]

print_matrix("RECURRENT + GATED : DISTANCE TO WEIGHTONLY", gain_grid, inhib_grid, rg_dist_map)
print_matrix("RECURRENT + GATED : SYNERGY INDEX", gain_grid, inhib_grid, rg_syn_map)
print_matrix("RECURRENT + GATED : NEAREST PARENT", gain_grid, inhib_grid, rg_parent_map)


# ================== 打印 top-k 协同点 ==================
def top_k_from_map(name, row_values, col_values, syn_map, dist_map, k=8):
    flat = []
    for i, r in enumerate(row_values):
        for j, c in enumerate(col_values):
            syn_v = syn_map[i, j]
            dist_v = dist_map[i, j]
            if np.isnan(syn_v) or np.isnan(dist_v):
                continue
            flat.append((r, c, syn_v, dist_v))

    flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 120)
    print(f"TOP {k} SYNERGY POINTS: {name}")
    print("=" * 120)
    print(f"{'row':>12} {'col':>12} {'synergy':>12} {'dist_to_weight':>16} {'zone':>16}")
    print("-" * 120)
    for item in flat_sorted[:k]:
        zone = mode_zone(item[3])
        print(f"{str(item[0]):>12} {str(item[1]):>12} {item[2]:12.6f} {item[3]:16.6f} {zone:>16}")



# ================== 区域统计 ==================
def zone_stats(name, dist_map, syn_map):
    valid_dist = dist_map[~np.isnan(dist_map)]
    valid_syn = syn_map[~np.isnan(syn_map)]

    near = np.sum(valid_dist < 1.2)
    mid = np.sum((valid_dist >= 1.2) & (valid_dist < 3.0))
    far = np.sum(valid_dist >= 3.0)
    syn_pos = np.sum(valid_syn > 0)
    syn_strong = np.sum(valid_syn > 1.0)

    print("\n" + "=" * 120)
    print(f"ZONE / SYNERGY STATS: {name}")
    print("=" * 120)
    print(f"valid point count     : {len(valid_dist)}")
    print(f"near-baseline count   : {near}")
    print(f"mid-shift count       : {mid}")
    print(f"far-shift count       : {far}")
    print(f"synergy > 0 count     : {syn_pos}")
    print(f"synergy > 1 count     : {syn_strong}")
    print(f"max synergy           : {np.max(valid_syn):.6f}")
    print(f"mean synergy          : {np.mean(valid_syn):.6f}")
    print(f"max dist              : {np.max(valid_dist):.6f}")
    print(f"mean dist             : {np.mean(valid_dist):.6f}")

print("\nDone.")