import nengo
import numpy as np
import matplotlib.pyplot as plt

# 创建模型
model = nengo.Network(label="Mode Comparison")

with model:
    # 输入节点：模拟情感事件（一个脉冲）
    def pulse_generator(t):
        return 1.0 if 0.1 < t < 0.11 else 0.0
    
    stimulus = nengo.Node(pulse_generator)
    
    # 中间神经元群体（用于循环模式）
    ens = nengo.Ensemble(100, dimensions=1)
    
    # 抑制性中间神经元（用于门控模式）
    inhibitory = nengo.Ensemble(50, 1)
    
    # 四个输出节点
    output0 = nengo.Node(size_in=1)  # 直接模式
    output1 = nengo.Node(size_in=1)  # 延迟模式
    output2 = nengo.Node(size_in=1)  # 循环模式
    output3 = nengo.Node(size_in=1)  # 门控模式
    
    # ---- 模式0：直接连接（基模）----
    nengo.Connection(stimulus, output0, synapse=0.001, transform=1.0)
    
    # ---- 模式1：延迟连接（延迟模）----
    nengo.Connection(stimulus, output1, synapse=0.5, transform=1.0)
    
    # ---- 模式2：循环连接 ----
    nengo.Connection(stimulus, ens, synapse=0.001, transform=1.0)
    nengo.Connection(ens, ens, synapse=0.05, transform=0.99)  # 反馈
    nengo.Connection(ens, output2, synapse=0.001, transform=1.0)
    
    # ---- 模式3：门控模式（带抑制性中间神经元）----
    # 兴奋路径（直接）
    nengo.Connection(stimulus, output3, synapse=0.001, transform=0.5)
    # 抑制路径（通过抑制性中间神经元）
    nengo.Connection(stimulus, inhibitory, synapse=0.05, transform=1.0)
    nengo.Connection(inhibitory, output3, synapse=0.01, transform=-1.5)
    
    # 记录探针
    p0 = nengo.Probe(output0, synapse=0.001)
    p1 = nengo.Probe(output1, synapse=0.001)
    p2 = nengo.Probe(output2, synapse=0.001)
    p3 = nengo.Probe(output3, synapse=0.001)

# 运行模拟
print("Running simulation...")
with nengo.Simulator(model) as sim:
    sim.run(2.0)

# 绘制结果（2x2子图）
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

modes = [
    (p0, 'Direct (M0)', 'blue'),
    (p1, 'Delayed (M1)', 'red'),
    (p2, 'Recurrent (M2)', 'green'),
    (p3, 'Gated (M3)', 'purple')
]

for i, (probe, title, color) in enumerate(modes):
    axes[i].plot(sim.trange(), sim.data[probe], color, linewidth=2)
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Emotional Intensity')
    axes[i].set_title(title)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()
input("\n按回车退出...")