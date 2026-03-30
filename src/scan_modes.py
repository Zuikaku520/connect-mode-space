import nengo
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# ================== 参数设置 ==================
SIM_TIME = 2.0          # 仿真时长 (s)
PULSE_START = 0.1
PULSE_END = 0.11

# 定义需要扫描的模式及其参数范围
scans = []

# 模式1：延迟模式 (synapse)
delays = [0.001, 0.05, 0.1, 0.2, 0.5, 1.0]
scans.append(('Delayed', 'synapse', delays, {}))

# 模式2：循环模式 (反馈增益 g)
gains = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
scans.append(('Recurrent', 'gain', gains, {}))

# 模式3：门控模式 (抑制强度 inhib_strength)
inhib_strengths = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5]
scans.append(('Gated', 'inhib_strength', inhib_strengths, {}))

# ================== 辅助函数 ==================
def ensure_1d(signal):
    """确保信号是一维数组"""
    signal = np.array(signal)
    if signal.ndim > 1:
        signal = signal.flatten()
    return signal

def extract_metrics(time, signal):
    """从时间序列中提取量化指标"""
    # 确保是一维
    signal = ensure_1d(signal)
    time = ensure_1d(time)
    
    # 找到脉冲后的信号 (从0.1s之后)
    mask = time > PULSE_START
    t_after = time[mask]
    s_after = signal[mask]
    
    if len(s_after) == 0:
        return None
    
    # 峰值检测 - 使用更简单的方法避免scipy问题
    # 直接找到最大值位置
    peak_idx = np.argmax(s_after)
    peak_time = t_after[peak_idx]
    peak_value = s_after[peak_idx]
    
    # 如果峰值太小，可能没有有效信号
    if peak_value < 0.01:
        return {
            'peak_time': np.nan,
            'half_life': np.nan,
            'overshoot': np.nan,
            'undershoot': 0.0,
            'peak_value': peak_value,
            'steady_val': 0.0
        }
    
    # 半衰期: 从峰值下降到峰值一半的时间
    half_val = peak_value / 2.0
    after_peak = s_after[peak_idx:]
    t_after_peak = t_after[peak_idx:]
    
    # 找到第一个 <= half_val 的点
    idx_half = np.where(after_peak <= half_val)[0]
    if len(idx_half) == 0:
        half_life = np.nan
    else:
        half_time = t_after_peak[idx_half[0]]
        half_life = half_time - peak_time
    
    # 稳态值: 最后0.5秒的平均
    steady_mask = time > (SIM_TIME - 0.5)
    steady_val = np.mean(signal[steady_mask])
    
    # 过冲率
    if steady_val != 0:
        overshoot = (peak_value - steady_val) / abs(steady_val)
    else:
        overshoot = np.nan
    
    # 欠冲深度: 最小负值
    undershoot = np.min(signal) if np.min(signal) < 0 else 0.0
    
    return {
        'peak_time': peak_time,
        'half_life': half_life,
        'overshoot': overshoot,
        'undershoot': undershoot,
        'peak_value': peak_value,
        'steady_val': steady_val
    }

def run_single_mode(mode_name, param_name, param_val, fixed_params):
    """运行单一模式的一次仿真，返回输出信号和指标"""
    with nengo.Network() as model:
        def pulse_generator(t):
            return 1.0 if PULSE_START < t < PULSE_END else 0.0
        stimulus = nengo.Node(pulse_generator)
        
        # 根据模式构建网络
        if mode_name == 'Delayed':
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
            # 兴奋路径
            nengo.Connection(stimulus, output, synapse=0.001, transform=0.5)
            # 抑制路径
            nengo.Connection(stimulus, inhibitory, synapse=0.05, transform=1.0)
            nengo.Connection(inhibitory, output, synapse=0.01, transform=param_val)
            probe = nengo.Probe(output, synapse=0.001)
        
        else:
            raise ValueError(f"Unknown mode: {mode_name}")
        
        with nengo.Simulator(model) as sim:
            sim.run(SIM_TIME)
    
    time = sim.trange()
    signal = sim.data[probe]
    metrics = extract_metrics(time, signal)
    return time, signal, metrics

# ================== 运行基线直接模式 ==================
print("Running baseline: Direct mode...")
with nengo.Network() as model:
    def pulse_generator(t):
        return 1.0 if PULSE_START < t < PULSE_END else 0.0
    stimulus = nengo.Node(pulse_generator)
    output = nengo.Node(size_in=1)
    nengo.Connection(stimulus, output, synapse=0.001, transform=1.0)
    probe = nengo.Probe(output, synapse=0.001)
    with nengo.Simulator(model) as sim:
        sim.run(SIM_TIME)
direct_time = sim.trange()
direct_signal = sim.data[probe]
direct_metrics = extract_metrics(direct_time, direct_signal)
print("Baseline done.\n")

# ================== 扫描所有模式 ==================
results = {}

for mode_name, param_name, param_vals, fixed in scans:
    print(f"Scanning {mode_name} mode over {param_name} = {param_vals}")
    results[mode_name] = {
        'param_name': param_name,
        'param_vals': [],
        'metrics': []
    }
    for val in param_vals:
        print(f"  Running {param_name}={val}...")
        try:
            _, _, metrics = run_single_mode(mode_name, param_name, val, fixed)
            if metrics is not None and not np.isnan(metrics['peak_time']):
                results[mode_name]['param_vals'].append(val)
                results[mode_name]['metrics'].append(metrics)
                print(f"    peak_time={metrics['peak_time']:.3f}s, half_life={metrics['half_life']:.3f}s")
            else:
                print(f"    No valid peak detected")
        except Exception as e:
            print(f"    Error: {e}")
    print()

# ================== 可视化：指标随参数变化 ==================
metrics_keys = ['peak_time', 'half_life', 'overshoot', 'undershoot']
metric_labels = {
    'peak_time': 'Peak Time (s)',
    'half_life': 'Half-life (s)',
    'overshoot': 'Overshoot Ratio',
    'undershoot': 'Undershoot Depth'
}

for mode_name, data in results.items():
    if len(data['param_vals']) == 0:
        print(f"No valid data for {mode_name}")
        continue
    
    param_vals = data['param_vals']
    param_name = data['param_name']
    
    # 收集各指标列表
    metric_lists = {k: [] for k in metrics_keys}
    for m in data['metrics']:
        for k in metrics_keys:
            metric_lists[k].append(m[k])
    
    # 绘制子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, key in enumerate(metrics_keys):
        ax = axes[idx]
        yvals = np.array(metric_lists[key])
        
        # 过滤nan
        valid_mask = ~np.isnan(yvals)
        if np.sum(valid_mask) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(f'{metric_labels[key]}')
            continue
        
        xvals = np.array(param_vals)[valid_mask]
        yvals_clean = yvals[valid_mask]
        
        ax.plot(xvals, yvals_clean, 'o-', color='b', linewidth=2, markersize=8)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric_labels[key])
        ax.set_title(f'{mode_name}: {metric_labels[key]}')
        ax.grid(True, alpha=0.3)
        
        # 计算相关系数
        if len(xvals) >= 3:
            r, p = pearsonr(xvals, yvals_clean)
            ax.text(0.7, 0.9, f'r = {r:.3f}\np = {p:.3f}', 
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{mode_name}_metrics.png', dpi=150)
    plt.show()
    print(f"Saved {mode_name}_metrics.png")

# ================== 输出统计表格 ==================
print("\n" + "="*70)
print("STATISTICAL SUMMARY (Pearson correlation)")
print("="*70)

all_results = []

for mode_name, data in results.items():
    if len(data['param_vals']) == 0:
        continue
    
    param_vals = data['param_vals']
    param_name = data['param_name']
    
    print(f"\n{mode_name} mode: {param_name} vs metrics")
    print("-" * 50)
    
    for key in metrics_keys:
        yvals = [m[key] for m in data['metrics']]
        valid_mask = ~np.isnan(yvals)
        
        if np.sum(valid_mask) < 2:
            print(f"  {key:15s}: insufficient data")
            continue
        
        xv = np.array(param_vals)[valid_mask]
        yv = np.array(yvals)[valid_mask]
        r, p = pearsonr(xv, yv)
        
        # 判断相关性强度
        strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
        sig = "significant" if p < 0.05 else "not significant"
        
        print(f"  {key:15s}: r = {r:6.3f}  ({strength}), p = {p:.4f} ({sig})")
        
        all_results.append({
            'mode': mode_name,
            'metric': key,
            'r': r,
            'p': p,
            'strength': strength,
            'significant': p < 0.05
        })

# 基线结果
print("\n" + "="*70)
print("BASELINE (Direct Mode) Metrics")
print("="*70)
for k in metrics_keys:
    val = direct_metrics[k]
    if val is not None and not np.isnan(val):
        print(f"  {k:15s}: {val:.4f}")
    else:
        print(f"  {k:15s}: N/A")

# 总结
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
significant_findings = [r for r in all_results if r['significant']]
print(f"Total significant correlations found: {len(significant_findings)}")

if len(significant_findings) > 0:
    print("\nKey findings:")
    for r in significant_findings:
        if abs(r['r']) > 0.7:
            print(f"  • {r['mode']} mode: {r['metric']} is strongly correlated with parameter (r={r['r']:.3f})")

print("\n" + "="*70)