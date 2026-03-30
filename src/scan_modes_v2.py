import nengo
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ================== 参数设置 ==================
SIM_TIME = 2.0
DT = 0.001
PULSE_START = 0.1
PULSE_END = 0.11

# 参数扫描
scans = []

# 基础对照：只调权重，不改连接模式
weights = np.linspace(0.2, 2.0, 10)
scans.append(('WeightOnly', 'weight', weights, {}))

# 延迟/滤波模式
delays = np.array([0.001, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5, 0.8])
scans.append(('Delayed', 'synapse', delays, {}))

# 反馈模式
gains = np.array([0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.9, 0.94, 0.97, 0.99])
scans.append(('Recurrent', 'gain', gains, {}))

# 门控模式
inhib_strengths = np.array([0.0, -0.2, -0.5, -0.8, -1.1, -1.4, -1.7, -2.0, -2.3, -2.6])
scans.append(('Gated', 'inhib_strength', inhib_strengths, {}))


# ================== 辅助函数 ==================
def ensure_1d(signal):
    signal = np.array(signal)
    if signal.ndim > 1:
        signal = signal.flatten()
    return signal


def pulse_generator(t):
    return 1.0 if PULSE_START < t < PULSE_END else 0.0


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
            'overshoot': np.nan,
            'undershoot': 0.0,
            'peak_value': peak_value,
            'steady_val': 0.0,
            'auc': np.nan,
            'recovery_slope': np.nan,
            'settling_time': np.nan
        }

    # 半衰期
    half_val = peak_value / 2.0
    after_peak = s_after[peak_idx:]
    t_after_peak = t_after[peak_idx:]
    idx_half = np.where(after_peak <= half_val)[0]
    if len(idx_half) == 0:
        half_life = np.nan
    else:
        half_life = t_after_peak[idx_half[0]] - peak_time

    # 稳态值
    steady_mask = time > (SIM_TIME - 0.3)
    steady_val = np.mean(signal[steady_mask])

    # 过冲
    if np.abs(steady_val) > 1e-8:
        overshoot = (peak_value - steady_val) / np.abs(steady_val)
    else:
        overshoot = np.nan

    # 欠冲
    undershoot = np.min(signal) if np.min(signal) < 0 else 0.0

    # AUC
    auc = np.trapz(np.abs(s_after), t_after)

    # 恢复斜率：峰后 50ms
    recovery_window = (t_after >= peak_time) & (t_after <= min(peak_time + 0.05, SIM_TIME))
    if np.sum(recovery_window) >= 3:
        x = t_after[recovery_window]
        y = s_after[recovery_window]
        coef = np.polyfit(x, y, 1)
        recovery_slope = coef[0]
    else:
        recovery_slope = np.nan

    # 稳定时间
    tol = max(0.05 * np.abs(peak_value), 1e-4)
    settling_time = np.nan
    for i in range(peak_idx, len(s_after)):
        if np.all(np.abs(s_after[i:] - steady_val) < tol):
            settling_time = t_after[i] - peak_time
            break

    return {
        'peak_time': peak_time,
        'half_life': half_life,
        'overshoot': overshoot,
        'undershoot': undershoot,
        'peak_value': peak_value,
        'steady_val': steady_val,
        'auc': auc,
        'recovery_slope': recovery_slope,
        'settling_time': settling_time
    }


def print_metric_table(mode_name, data, feature_keys):
    print("\n" + "=" * 100)
    print(f"{mode_name} FEATURE TABLE")
    print("=" * 100)

    header = f"{'param':>10} " + " ".join([f"{k:>15}" for k in feature_keys])
    print(header)
    print("-" * 100)

    for p, m in zip(data['param_vals'], data['metrics']):
        vals = []
        for k in feature_keys:
            v = m[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(f"{'nan':>15}")
            else:
                vals.append(f"{v:15.6f}")
        print(f"{p:10.4f} " + " ".join(vals))


def run_single_mode(mode_name, param_val):
    with nengo.Network() as model:
        stimulus = nengo.Node(pulse_generator)

        if mode_name == 'WeightOnly':
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, output, synapse=0.001, transform=param_val)
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Delayed':
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, output, synapse=param_val, transform=1.0)
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Recurrent':
            ens = nengo.Ensemble(100, dimensions=1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens, synapse=0.001, transform=1.0)
            nengo.Connection(ens, ens, synapse=0.05, transform=param_val)
            nengo.Connection(ens, output, synapse=0.001, transform=1.0)
            probe = nengo.Probe(output, synapse=0.001)

        elif mode_name == 'Gated':
            inhibitory = nengo.Ensemble(50, 1)
            output = nengo.Node(size_in=1)
            nengo.Connection(stimulus, output, synapse=0.001, transform=0.5)
            nengo.Connection(stimulus, inhibitory, synapse=0.05, transform=1.0)
            nengo.Connection(inhibitory, output, synapse=0.01, transform=param_val)
            probe = nengo.Probe(output, synapse=0.001)

        else:
            raise ValueError(f"Unknown mode: {mode_name}")

        with nengo.Simulator(model, dt=DT, progress_bar=False) as sim:
            sim.run(SIM_TIME)

    time = sim.trange()
    signal = sim.data[probe]
    metrics = extract_metrics(time, signal)
    return time, signal, metrics


def build_feature_matrix(results, mode_name, feature_keys):
    rows = []
    params = []
    for p, m in zip(results[mode_name]['param_vals'], results[mode_name]['metrics']):
        vec = [m[k] for k in feature_keys]
        if not np.any(np.isnan(vec)):
            rows.append(vec)
            params.append(p)
    return np.array(rows, dtype=float), np.array(params, dtype=float)


# ================== 基线 Direct ==================
print("Running direct baseline...")
with nengo.Network() as model:
    stimulus = nengo.Node(pulse_generator)
    output = nengo.Node(size_in=1)
    nengo.Connection(stimulus, output, synapse=0.001, transform=1.0)
    probe = nengo.Probe(output, synapse=0.001)
    with nengo.Simulator(model, dt=DT, progress_bar=False) as sim:
        sim.run(SIM_TIME)

direct_time = sim.trange()
direct_signal = sim.data[probe]
direct_metrics = extract_metrics(direct_time, direct_signal)
print("Direct baseline done.\n")


# ================== 扫描 ==================
results = {}
waveforms = {}

for mode_name, param_name, param_vals, _ in scans:
    print(f"Scanning {mode_name} over {param_name}...")
    results[mode_name] = {
        'param_name': param_name,
        'param_vals': [],
        'metrics': []
    }
    waveforms[mode_name] = []

    for val in param_vals:
        try:
            t, s, m = run_single_mode(mode_name, val)
            if m is not None and not np.isnan(m['peak_time']):
                results[mode_name]['param_vals'].append(val)
                results[mode_name]['metrics'].append(m)
                waveforms[mode_name].append((val, t, s))
                print(f"  {param_name}={val:.4f} -> peak_time={m['peak_time']:.4f}, half_life={m['half_life']}")
            else:
                print(f"  {param_name}={val:.4f} -> invalid")
        except Exception as e:
            print(f"  {param_name}={val:.4f} -> ERROR: {e}")
    print()


# ================== 每个模式详细表 ==================
table_keys = [
    'peak_time', 'half_life', 'undershoot',
    'auc', 'recovery_slope', 'settling_time', 'peak_value'
]

for mode_name, data in results.items():
    print_metric_table(mode_name, data, table_keys)


# ================== 相关性输出 ==================
metrics_keys = [
    'peak_time', 'half_life', 'overshoot', 'undershoot',
    'peak_value', 'auc', 'recovery_slope', 'settling_time'
]

print("\n" + "=" * 100)
print("STATISTICAL SUMMARY")
print("=" * 100)

for mode_name, data in results.items():
    print(f"\n{mode_name}")
    print("-" * 100)
    x = np.array(data['param_vals'])
    for key in metrics_keys:
        y = np.array([m[key] for m in data['metrics']], dtype=float)
        valid = ~np.isnan(y)
        if np.sum(valid) >= 3:
            r, p = pearsonr(x[valid], y[valid])
            strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
            sig = "significant" if p < 0.05 else "not significant"
            print(f"{key:15s}: r={r:8.3f}, p={p:.6f} -> {strength}, {sig}")
        else:
            print(f"{key:15s}: insufficient data")


# ================== 模式替代测试 ==================
print("\n" + "=" * 100)
print("MODE REPLACEMENT TEST")
print("=" * 100)

feature_keys = ['peak_time', 'half_life', 'undershoot', 'auc', 'recovery_slope', 'settling_time']

weight_mat, weight_params = build_feature_matrix(results, 'WeightOnly', feature_keys)

replacement_summary = {}

for mode_name in ['Delayed', 'Recurrent', 'Gated']:
    target_mat, target_params = build_feature_matrix(results, mode_name, feature_keys)
    if len(weight_mat) == 0 or len(target_mat) == 0:
        print(f"{mode_name}: insufficient comparable data")
        continue

    scaler = StandardScaler()
    all_mat = np.vstack([weight_mat, target_mat])
    all_scaled = scaler.fit_transform(all_mat)

    w_scaled = all_scaled[:len(weight_mat)]
    t_scaled = all_scaled[len(weight_mat):]

    dist = cdist(t_scaled, w_scaled, metric='euclidean')
    nearest_idx = np.argmin(dist, axis=1)
    nearest_dist = np.min(dist, axis=1)

    replacement_summary[mode_name] = {
        'mean_dist': np.mean(nearest_dist),
        'std_dist': np.std(nearest_dist),
        'max_dist': np.max(nearest_dist),
        'min_dist': np.min(nearest_dist)
    }

    print(f"\n{mode_name} vs WeightOnly")
    print("-" * 100)
    print(f"{'target_param':>12} {'best_weight':>12} {'distance':>14}")
    print("-" * 100)
    for i in range(len(target_params)):
        print(f"{target_params[i]:12.4f} {weight_params[nearest_idx[i]]:12.4f} {nearest_dist[i]:14.6f}")

    print("-" * 100)
    print(f"mean nearest distance = {np.mean(nearest_dist):.6f}")
    print(f"std  nearest distance = {np.std(nearest_dist):.6f}")
    print(f"max  nearest distance = {np.max(nearest_dist):.6f}")
    print(f"min  nearest distance = {np.min(nearest_dist):.6f}")


# ================== 聚类与 PCA ==================
print("\n" + "=" * 100)
print("CLUSTERING / PCA")
print("=" * 100)

all_rows = []
all_labels = []
all_params = []

for mode_name in results.keys():
    for p, m in zip(results[mode_name]['param_vals'], results[mode_name]['metrics']):
        vec = [m[k] for k in feature_keys]
        if not np.any(np.isnan(vec)):
            all_rows.append(vec)
            all_labels.append(mode_name)
            all_params.append(p)

all_rows = np.array(all_rows, dtype=float)

if len(all_rows) >= 4:
    scaler = StandardScaler()
    X = scaler.fit_transform(all_rows)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    print("\n" + "=" * 100)
    print("PCA COORDINATES")
    print("=" * 100)
    print(f"{'mode':>12} {'param':>12} {'PC1':>14} {'PC2':>14} {'cluster':>10}")
    print("-" * 100)
    for i in range(len(all_labels)):
        print(f"{all_labels[i]:>12} {all_params[i]:12.4f} {X2[i,0]:14.6f} {X2[i,1]:14.6f} {clusters[i]:10d}")

    # 仍保留图片保存，但你不看也没事
    plt.figure(figsize=(9, 7))
    markers = {
        'WeightOnly': 'o',
        'Delayed': 's',
        'Recurrent': '^',
        'Gated': 'D'
    }
    for mode_name in markers.keys():
        idx = [i for i, lab in enumerate(all_labels) if lab == mode_name]
        if len(idx) > 0:
            plt.scatter(X2[idx, 0], X2[idx, 1], marker=markers[mode_name], label=mode_name, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of dynamical features by connection mode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mode_pca.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        plt.scatter(X2[idx, 0], X2[idx, 1], label=f"Cluster {c}", alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans clusters in dynamical feature space")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mode_clusters.png", dpi=150)
    plt.close()
else:
    print("Insufficient data for PCA / clustering.")


# ================== 波形图保存 ==================
plt.figure(figsize=(12, 8))
for mode_name in ['WeightOnly', 'Delayed', 'Recurrent', 'Gated']:
    if len(waveforms[mode_name]) == 0:
        continue
    chosen = [
        waveforms[mode_name][0],
        waveforms[mode_name][len(waveforms[mode_name]) // 2],
        waveforms[mode_name][-1]
    ]
    for val, t, s in chosen:
        plt.plot(t, s, label=f"{mode_name}:{val:.3f}")

plt.axvspan(PULSE_START, PULSE_END, alpha=0.15)
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Representative waveforms across connection modes")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("representative_waveforms.png", dpi=150)
plt.close()


# ================== 模式均值摘要 ==================
print("\n" + "=" * 100)
print("MODE FEATURE MEAN SUMMARY")
print("=" * 100)

for mode_name in results.keys():
    rows = []
    for m in results[mode_name]['metrics']:
        vec = [m[k] for k in feature_keys]
        if not np.any(np.isnan(vec)):
            rows.append(vec)

    if len(rows) == 0:
        print(f"\n{mode_name}: no valid rows")
        continue

    rows = np.array(rows, dtype=float)
    mean_vals = np.mean(rows, axis=0)
    std_vals = np.std(rows, axis=0)

    print(f"\n{mode_name}")
    print("-" * 70)
    for k, mean_v, std_v in zip(feature_keys, mean_vals, std_vals):
        print(f"{k:15s}: mean={mean_v:12.6f}, std={std_v:12.6f}")


# ================== Direct baseline ==================
print("\n" + "=" * 100)
print("DIRECT BASELINE")
print("=" * 100)
for k, v in direct_metrics.items():
    if isinstance(v, float) and not np.isnan(v):
        print(f"{k:15s}: {v:.6f}")
    else:
        print(f"{k:15s}: {v}")


# ================== 自动结论 ==================
print("\n" + "=" * 100)
print("AUTO INTERPRETATION")
print("=" * 100)

for mode_name in ['Delayed', 'Recurrent', 'Gated']:
    if mode_name not in replacement_summary:
        continue

    mean_dist = replacement_summary[mode_name]['mean_dist']

    if mean_dist > 1.5:
        verdict = "strongly NOT replaceable by weight-only"
    elif mean_dist > 0.8:
        verdict = "partially not replaceable by weight-only"
    else:
        verdict = "may be partly approximated by weight-only"

    print(f"{mode_name:12s}: mean distance = {mean_dist:.6f} -> {verdict}")

print("\nDone.")