# Connect Mode Space
### From common line to connection mode dynamics

## English Summary

This repository explores the hypothesis that connection mode should be treated as an independent dynamical variable in neural computation, rather than being reduced to a common weighted line.

Most neural models focus on nodes and weights.
This project asks a different question:

What if the connection itself is not just a line?
What if different connection modes produce fundamentally different dynamical behaviors?

The starting intuition came from optical fiber mode theory:
in optics, propagation cannot be reduced to a simple transmission pipe.
This repository explores whether a similar under-modeling may exist in neural-style computation.


## Current Findings

- Connection mode cannot be ignored.
- Connection mode is not reducible to weight-only adjustment.
- Higher-order combinations can push the system into new dynamical regions.
- Synergy is not uniform; it emerges only in localized parameter windows.
- In current experiments, `Delayed + Recurrent` shows the clearest stable synergy hotspots.
- Hotspot regions and control regions exhibit clearly different synergy statistics.

## Project Status

This is an exploratory computational research project.
It is not a biological proof.
It is not a proof of consciousness.
It is currently a simulation-based hypothesis with progressively stronger empirical support.

This repository does not claim biological proof or proof of consciousness.
It currently presents a computational hypothesis supported by progressively stronger simulation evidence.

## How to use？

```pip install -r requirements.txt```

```python src/scan_modes_v8.py```

## Experiment Evolution
- `scan_modes.py`: minimal mode verification
- `scan_modes_v2.py`: weight-only replacement testing
- `scan_modes_v3.py`: combined modes and early mode-space analysis
- `scan_modes_v4.py`: robustness under repeated noisy runs
- `scan_modes_v6.py`: synergy index, ablation, and incremental contribution
- `scan_modes_v7.py`: 2D parameter-space mapping
- `scan_modes_v8.py`: local hotspot refinement

## Version Index

| File | Main Role |
|------|-----------|
| `scan_modes.py` | Minimal single-mode verification |
| `scan_modes_v2.py` | Weight-only replacement test |
| `scan_modes_v3.py` | Combined modes and early mode-space analysis |
| `scan_modes_v4.py` | Robustness under repeated noisy runs |
| `scan_modes_v6.py` | Synergy index, ablation, incremental contribution |
| `scan_modes_v7.py` | 2D mode-space mapping |
| `scan_modes_v8.py` | Local hotspot refinement |

---

## 中文说明

## 项目简介
本仓库探索一个假设：

在神经计算中，节点之间的连接不应被统一等效为 common line，
而应被视为具有独立动力学属性的 connection mode。

本项目最初受到光纤模式传播问题的启发：
在光学系统中，传播过程不能被简化为普通传输管道。
受此启发，本项目尝试检验：在神经风格计算中，连接本身是否也可能是一个被长期低估的建模层。

## 当前结果
目前的最小仿真结果支持以下判断：

- 连接模式不能被忽略
- 连接模式不能被单纯 weight-only 调节替代
- 高阶组合会将系统推进到新的动力学区域
- 协同不是到处都有，而是存在局部热点窗口
- 在当前实验中，Delayed + Recurrent 显示出最明显的协同热点区

## 当前状态
这是一个探索性研究仓库，不是论文成品，也不是生物学证明。
当前内容更适合被理解为：

- 一个计算研究假设
- 一组逐步增强的仿真实验
- 一套关于 connection mode space 的初步理论框架

## 仓库结构
- `src/`：代码版本演化
- `docs/`：理论、实验日志与结论整理
- `results/`：原始输出和阶段总结

## 如何运行
```bash
pip install -r requirements.txt
python src/scan_modes_v8.py
