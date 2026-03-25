# Kalman Filter — Engineering Notes

## Architecture Decision: Why Three Variants?

Standard KF is fine for linear systems. But real sensor fusion is never linear:
- GPS/INS integration: attitude quaternion creates nonlinear state transition
- Radar tracking: polar measurements (range, bearing) vs Cartesian state
- Relative navigation: rotation matrices in state vector

**EKF choice:** Linearize at current estimate. Works well when nonlinearity is mild
and you have analytical Jacobians. Computing Jacobians numerically (finite differences)
is fragile — a 0.01 perturbation that's fine for position goes wrong for angles that
wrap at ±π.

**UKF choice:** Use sigma points to propagate uncertainty through nonlinearities exactly
to third order. Costs 2n+1 model evaluations per step vs 1 for EKF. Worth it when:
- EKF diverges due to high nonlinearity
- You can't derive Jacobians analytically
- State includes quaternions (UKF handles SO(3) naturally with the right sigma point scheme)

The tradeoff nobody talks about: UKF is harder to debug. EKF divergence is usually
traceable to a bad Jacobian. UKF divergence means either your sigma point weights are
wrong, your process model is bad, or you have numerical issues in the Cholesky.

## Numerical Stability

The Joseph form for covariance update:
```
P = (I - KH)P(I - KH)ᵀ + KRKᵀ
```
is more expensive than the naive form `P = (I-KH)P` but guarantees P stays symmetric
and positive semi-definite even with finite-precision arithmetic. In float32, the naive
form can make P asymmetric after ~1000 updates, leading to Cholesky failure in UKF.

At high update rates (>500Hz), consider square-root forms (SRCKF/SRUKF) that work
with Cholesky factors directly and propagate them through square-root operations.
This keeps P positive definite by construction — never need the jitter hack.

## The Measurement Gating Problem

Mahalanobis gating is essential for robustness. Without it:
- A single GPS spoofing packet at +100m error will corrupt the state estimate
  for 10+ seconds (depends on R/Q ratio)
- Sensor dropouts followed by reconnect cause "jumps" that downstream systems
  (autopilots, guidance laws) amplify dangerously

The χ² threshold selection is a real engineering decision:
- Too tight (p=0.001): reject legitimate measurements during high dynamics
- Too loose (p=0.999): accept spoofed measurements, defeats the purpose

In practice: tight during hover/loiter, loose during high-G maneuvers, with
an anomaly counter that flags persistent rejections (sensor failure vs transient).

## This Approach Breaks Down When

1. **Multimodal uncertainty**: KF assumes Gaussian belief. After a long GPS outage,
   the true position distribution can be multimodal (several places the vehicle could be).
   Use a particle filter or IMM (Interacting Multiple Model) instead.

2. **Colored noise**: KF assumes white process and measurement noise. Vibration-induced
   IMU noise is highly colored (mechanical resonances). Pre-filter with notch filter
   or augment state with vibration harmonics.

3. **Unknown system model**: If F changes (different flight modes, configuration changes),
   consider adaptive KF or model-switching IMM.

## In Production

SpaceX Dragon's navigation system: 15-state EKF running at 200Hz in C on a RAD750
processor. Backup: 3-state KF on independent hardware. GPS dropout during plasma
blackout handled by tight IMU integration with bias tracking. Star tracker provides
attitude resets on long burns.

The most important engineering decision isn't the filter type — it's sensor quality.
A well-calibrated IMU with basic KF outperforms a poor IMU with UKF.
