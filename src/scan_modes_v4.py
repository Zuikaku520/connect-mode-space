import nengo
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ================== 参数设置 ==================
SIM_TIME = 2.5
DT = 0.001
PULSE_START = 0.1
PULSE_END = 0.11

# 每个模式扫描点
weights = np.linspace(0.2, 2.0, 10)
delays = np.array([0.001, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5, 0.8])
gains = np.array([0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.9, 0.94, 0.97, 0.99])
inhib_strengths = np.array([0.0, -0.2, -0.5, -0.8, -1.1, -1.4, -1.7, -2.0, -2.3, -2.6])

# 固定一组组合模式参数
combo_delay_vals = np.array([0.01, 0.05, 0.12, 0.35, 0.8])
combo_gain_vals = np.array([0.2, 0.6, 0.85, 0.94, 0.99])
combo_inhib_vals = np.array([-0.2, -0.8, -1.4, -2.0, -2.6])

# 二元组合：全网格
delayed_recurrent_grid = [(d, g) for d in combo_delay_vals for g in combo_gain_vals]
delayed_gated_grid = [(d, inh) for d in combo_delay_vals for inh in combo_inhib_vals]
recurrent_gated_grid = [(g, inh) for g in combo_gain_vals for inh in combo_inhib_vals]

# 三元组合：先用稀疏网格，避免太慢
combo_delay_small = np.array([0.01, 0.12, 0.8])
combo_gain_small = np.array([0.2, 0.85, 0.99])
combo_inhib_small = np.array([-0.2, -1.4, -2.6])
triple_grid = [(d, g, inh) for d in combo_delay_small for g in combo_gain_small for inh in combo_inhib_small]

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

    # steady-state
    steady_mask = time > (SIM_TIME - 0.3)
    steady_val = np.mean(signal[steady_mask])

    # overshoot
    if np.abs(steady_val) > 1e-8:
        overshoot = (peak_value - steady_val) / np.abs(steady_val)
    else:
        overshoot = np.nan

    # undershoot
    undershoot = np.min(signal) if np.min(signal) < 0 else 0.0

    # auc
    auc = np.trapz(np.abs(s_after), t_after)

    # recovery slope (50ms after peak)
    recovery_window = (t_after >= peak_time) & (t_after <= min(peak_time + 0.05, SIM_TIME))
    if np.sum(recovery_window) >= 3:
        x = t_after[recovery_window]
        y = s_after[recovery_window]
        coef = np.polyfit(x, y, 1)
        recovery_slope = coef[0]
    else:
        recovery_slope = np.nan

    # settling time
    tol = max(0.05 * np.abs(peak_value), 1e-4)
    settling_time = np.nan
    for i in range(peak_idx, len(s_after)):
        if np.all(np.abs(s_after[i:] - steady_val) < tol):
            settling_time = t_after[i] - peak_time
            break

    # rise time (10% -> 90%)
    val10 = 0.1 * peak_value
    val90 = 0.9 * peak_value
    idx10 = np.where(s_after >= val10)[0]
    idx90 = np.where(s_after >= val90)[0]
    if len(idx10) > 0 and len(idx90) > 0:
        rise_time = t_after[idx90[0]] - t_after[idx10[0]]
    else:
        rise_time = np.nan

    # energy
    energy = np.trapz(s_after**2, t_after)

    return {
        'peak_time': peak_time,
        'half_life': half_life,
        'overshoot': overshoot,
        'undershoot': undershoot,
        'peak_value': peak_value,
        'steady_val': steady_val,
        'auc': auc,
        'recovery_slope': recovery_slope,
        'settling_time': settling_time,
        'rise_time': rise_time,
        'energy': energy
    }

def print_metric_table(mode_name, params, metrics, keys):
    print("\n" + "=" * 120)
    print(f"{mode_name} FEATURE TABLE")
    print("=" * 120)
    header = f"{'param':>18} " + " ".join([f"{k:>14}" for k in keys])
    print(header)
    print("-" * 120)
    for p, m in zip(params, metrics):
        row = []
        for k in keys:
            v = m[k]
            if isinstance(v, float) and np.isnan(v):
                row.append(f"{'nan':>14}")
            else:
                row.append(f"{v:14.6f}")
        print(f"{str(p):>18} " + " ".join(row))

# ================== 仿真函数 ==================
def run_single_mode(mode_name, param):
    with nengo.Network() as model:
        stimulus = nengo.Node(pulse_generator)

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
    metrics = extract_metrics(time, signal)
    return time, signal, metrics

# ================== 定义扫描空间 ==================
scan_dict = {
    'WeightOnly': list(weights),
    'Delayed': list(delays),
    'Recurrent': list(gains),
    'Gated': list(inhib_strengths),
    'Delayed+Recurrent':delayed_recurrent_grid,
    'Delayed+Gated': delayed_gated_grid,
    'Recurrent+Gated': recurrent_gated_grid,
    'Delayed+Recurrent+Gated': triple_grid,
}

# ================== 运行 Direct baseline ==================
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

# ================== 扫描所有模式 ==================
results = {}
waveforms = {}

for mode_name, param_list in scan_dict.items():
    print(f"Scanning {mode_name} ...")
    results[mode_name] = {'params': [], 'metrics': []}
    waveforms[mode_name] = []

    for p in param_list:
        try:
            t, s, m = run_single_mode(mode_name, p)
            if m is not None and not np.isnan(m['peak_time']):
                results[mode_name]['params'].append(p)
                results[mode_name]['metrics'].append(m)
                waveforms[mode_name].append((p, t, s))
                print(f"  param={p} -> peak_time={m['peak_time']:.4f}, auc={m['auc']:.6f}, settling={m['settling_time']}")
            else:
                print(f"  param={p} -> invalid")
        except Exception as e:
            print(f"  param={p} -> ERROR: {e}")
    print()

# ================== 打印详细表 ==================
table_keys = ['peak_time', 'half_life', 'undershoot', 'auc', 'recovery_slope', 'settling_time', 'rise_time', 'energy']
for mode_name, data in results.items():
    print_metric_table(mode_name, data['params'], data['metrics'], table_keys)

# ================== 统计摘要 ==================
metrics_keys = ['peak_time', 'half_life', 'overshoot', 'undershoot', 'peak_value', 'auc', 'recovery_slope', 'settling_time', 'rise_time', 'energy']

print("\n" + "=" * 120)
print("STATISTICAL SUMMARY")
print("=" * 120)

for mode_name, data in results.items():
    print(f"\n{mode_name}")
    print("-" * 120)
    # 对 tuple 参数，用索引号代替做 Pearson，避免维度问题
    x = np.arange(len(data['params']), dtype=float)
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

# ================== 构建特征矩阵 ==================
feature_keys = ['peak_time', 'half_life', 'undershoot', 'auc', 'recovery_slope', 'settling_time', 'rise_time', 'energy']

all_rows = []
all_labels = []
all_params = []

for mode_name, data in results.items():
    for p, m in zip(data['params'], data['metrics']):
        vec = [m[k] for k in feature_keys]
        if not np.any(np.isnan(vec)):
            all_rows.append(vec)
            all_labels.append(mode_name)
            all_params.append(p)

all_rows = np.array(all_rows, dtype=float)

# ================== WeightOnly 替代测试 ==================
print("\n" + "=" * 120)
print("MODE REPLACEMENT TEST AGAINST WEIGHTONLY")
print("=" * 120)

def build_feature_matrix(mode_name):
    rows, params = [], []
    for p, m in zip(results[mode_name]['params'], results[mode_name]['metrics']):
        vec = [m[k] for k in feature_keys]
        if not np.any(np.isnan(vec)):
            rows.append(vec)
            params.append(p)
    return np.array(rows, dtype=float), params

weight_mat, weight_params = build_feature_matrix('WeightOnly')
replacement_summary = {}

for mode_name in results.keys():
    if mode_name == 'WeightOnly':
        continue

    target_mat, target_params = build_feature_matrix(mode_name)
    if len(weight_mat) == 0 or len(target_mat) == 0:
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
        'min_dist': np.min(nearest_dist),
    }

    print(f"\n{mode_name} vs WeightOnly")
    print("-" * 120)
    print(f"{'target_param':>25} {'best_weight':>12} {'distance':>14}")
    print("-" * 120)
    for i in range(len(target_params)):
        print(f"{str(target_params[i]):>25} {weight_params[nearest_idx[i]]:12.4f} {nearest_dist[i]:14.6f}")
    print("-" * 120)
    print(f"mean nearest distance = {np.mean(nearest_dist):.6f}")
    print(f"std  nearest distance = {np.std(nearest_dist):.6f}")
    print(f"max  nearest distance = {np.max(nearest_dist):.6f}")
    print(f"min  nearest distance = {np.min(nearest_dist):.6f}")

# ================== PCA / 聚类 ==================
print("\n" + "=" * 120)
print("PCA / CLUSTERING")
print("=" * 120)

if len(all_rows) >= 8:
    scaler = StandardScaler()
    X = scaler.fit_transform(all_rows)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    n_clusters = min(8, len(np.unique(all_labels)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X)

    print("\n" + "=" * 120)
    print("PCA COORDINATES")
    print("=" * 120)
    print(f"{'mode':>28} {'param':>22} {'PC1':>14} {'PC2':>14} {'cluster':>10}")
    print("-" * 120)
    for i in range(len(all_labels)):
        print(f"{all_labels[i]:>28} {str(all_params[i]):>22} {X2[i,0]:14.6f} {X2[i,1]:14.6f} {clusters[i]:10d}")

# ================== 模式均值摘要 ==================
print("\n" + "=" * 120)
print("MODE FEATURE MEAN SUMMARY")
print("=" * 120)

for mode_name, data in results.items():
    rows = []
    for m in data['metrics']:
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
    print("-" * 80)
    for k, mean_v, std_v in zip(feature_keys, mean_vals, std_vals):
        print(f"{k:15s}: mean={mean_v:12.6f}, std={std_v:12.6f}")

# ================== 分类测试：能否从特征反推模式 ==================
print("\n" + "=" * 120)
print("MODE CLASSIFICATION TEST")
print("=" * 120)

valid_idx = []
for i, row in enumerate(all_rows):
    if not np.any(np.isnan(row)):
        valid_idx.append(i)

X = all_rows[valid_idx]
y_labels = [all_labels[i] for i in valid_idx]

if len(np.unique(y_labels)) >= 2 and len(X) >= 10:
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = np.min(class_counts)

    print("Class counts:")
    for cls, cnt in zip(unique_classes, class_counts):
        print(f"  class {cls} ({le.classes_[cls]}): {cnt}")

    # 样本太少的类别直接提醒
    rare_classes = unique_classes[class_counts < 2]
    if len(rare_classes) > 0:
        print("\n[WARNING] These classes have fewer than 2 samples:")
        for cls in rare_classes:
            print(f"  class {cls} ({le.classes_[cls]})")
        print("Falling back to non-stratified split.\n")

        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=0.3, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=0.3, random_state=42, stratify=y
        )

    # KNN 的邻居数不能超过最小训练类规模，简单保守一点
    k = min(3, len(X_train))
    if k < 1:
        print("Insufficient training samples for classification test.")
    else:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nClassification accuracy = {acc:.6f}")

        print("\nLabel mapping:")
        for i, name in enumerate(le.classes_):
            print(f"  {i} -> {name}")

        
        present_labels = np.unique(np.concatenate([y_test, y_pred]))
        present_names = [le.classes_[i] for i in present_labels]

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=present_labels))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=present_names,
            zero_division=0
        ))
else:
    print("Insufficient data for classification test.")

# ================== baseline ==================
print("\n" + "=" * 120)
print("DIRECT BASELINE")
print("=" * 120)
for k, v in direct_metrics.items():
    if isinstance(v, float) and not np.isnan(v):
        print(f"{k:15s}: {v:.6f}")
    else:
        print(f"{k:15s}: {v}")

# ================== 自动解释 ==================
print("\n" + "=" * 120)
print("AUTO INTERPRETATION")
print("=" * 120)

for mode_name, stat in replacement_summary.items():
    mean_dist = stat['mean_dist']
    if mean_dist > 2.5:
        verdict = "very strongly NOT replaceable by weight-only"
    elif mean_dist > 1.5:
        verdict = "strongly NOT replaceable by weight-only"
    elif mean_dist > 0.8:
        verdict = "partially not replaceable by weight-only"
    else:
        verdict = "may be partly approximated by weight-only"
    print(f"{mode_name:28s}: mean distance = {mean_dist:.6f} -> {verdict}")

print("\nDone.")