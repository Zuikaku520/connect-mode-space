# Connect Mode Space
### From common line to connection mode dynamics

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
- Synergy is not uniform; it emerges in localized windows.
- In current experiments, Delayed + Recurrent shows the clearest synergy hotspots.

## Project Status

This is an exploratory computational research project.
It is not a biological proof.
It is not a proof of consciousness.
It is currently a simulation-based hypothesis with progressively stronger empirical support.

##How to use？

```pip install -r requirements.txt```

```python src/scan_modes_v8.py```
