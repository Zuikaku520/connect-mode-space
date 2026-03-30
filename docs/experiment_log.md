## scan_modes.py
Question:
Do different single connection modes change output dynamics?

Result:
Yes. Delayed and Gated modes showed strong metric shifts.

## scan_modes_v2.py
Question:
Can these changes be approximated by weight-only tuning?

Result:
No. Multiple modes remained far from the WeightOnly baseline.

## scan_modes_v3.py
Question:
Do combined modes open new dynamical regions?

Result:
Yes. Combined modes were often farther from WeightOnly than single modes, suggesting the emergence of a mode space.

## scan_modes_v4.py
Question:
Are the effects robust under repeated noisy runs?

Result:
Yes. Key differences remained under repeated runs with input noise.

## scan_modes_v6.py
Question:
Do combined modes exhibit structured ablation patterns and measurable synergy?

Result:
Yes. Some combinations remained parent-dominated, while others showed stronger independence and positive synergy.

## scan_modes_v7.py
Question:
Does synergy appear uniformly across parameter space?

Result:
No. Different mode planes showed different structures. Delayed + Recurrent displayed the strongest synergy regions.

## scan_modes_v8.py
Question:
Are previously observed synergy hotspots isolated points or stable local windows?

Result:
Stable local windows were found. Hotspot A and Hotspot B showed strong local synergy, while the control region did not. :contentReference[oaicite:1]{index=1}