// FILE: aerospace/kalman-filter/tests/test_cases.js
//
// Test suite for Kalman filter implementations.
// Mix of deterministic unit tests and stochastic validation (Monte Carlo).
// "All models are wrong; some are useful." — Box
// A filter is "correct" if it's statistically consistent: the actual errors
// match the predicted covariance. NEES (Normalized Estimation Error Squared)
// is the gold standard test for this.

'use strict';

// Simplified test framework (no dependencies)
let passed = 0, failed = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch(e) {
    console.error(`  ✗ ${name}: ${e.message}`);
    failed++;
  }
}
function assert(condition, msg) { if (!condition) throw new Error(msg || 'Assertion failed'); }
function assertClose(a, b, tol = 1e-6, msg) {
  assert(Math.abs(a - b) <= tol, `${msg || ''}: expected ${a} ≈ ${b} (tol ${tol}), diff = ${Math.abs(a-b)}`);
}

// ─── KF Unit Tests ────────────────────────────────────────────────────────────

import { KalmanFilter } from '../src/implementation.js';

console.log('\n── Linear Kalman Filter ──');

test('Constant position: filter should converge to true value', () => {
  // 1D constant position: x[k] = x[k-1], z[k] = x[k] + noise
  const n = 1, m = 1;
  const F = new Float64Array([1]);
  const H = new Float64Array([1]);
  const Q = new Float64Array([1e-4]);   // very small process noise
  const R = new Float64Array([1.0]);    // measurement noise variance
  const x0 = new Float64Array([0]);
  const P0 = new Float64Array([10]);    // large initial uncertainty

  const kf = new KalmanFilter({ n, m, F, H, Q, R, x0, P0 });
  const TRUE_VALUE = 5.0;

  for (let i = 0; i < 100; i++) {
    kf.predict();
    const z = new Float64Array([TRUE_VALUE + (Math.random()-0.5) * 2]);
    kf.update(z);
  }

  // After 100 measurements, estimate should be close
  assertClose(kf.state[0], TRUE_VALUE, 0.5, 'Estimate should converge');
  // Posterior variance should be much smaller than prior
  assert(kf.covariance[0] < 0.1, 'Posterior variance should decrease significantly');
});

test('Constant velocity: position and velocity estimates', () => {
  // 2D state: [pos, vel], transition: pos += vel*dt, vel = const
  const dt = 0.1, n = 2, m = 1;
  const F = new Float64Array([1, dt, 0, 1]);
  const H = new Float64Array([1, 0]);
  const Q = new Float64Array([dt**4/4, dt**3/2, dt**3/2, dt**2].map(v => v * 0.01));
  const R = new Float64Array([0.5]);
  const x0 = new Float64Array([0, 0]);
  const P0 = new Float64Array([100, 0, 0, 100]);

  const kf = new KalmanFilter({ n, m, F, H, Q, R, x0, P0 });

  const TRUE_VEL = 2.0;
  let truePos = 0;

  for (let i = 0; i < 200; i++) {
    truePos += TRUE_VEL * dt;
    kf.predict();
    const z = new Float64Array([truePos + (Math.random()-0.5) * 2]);
    kf.update(z);
  }

  const [estPos, estVel] = kf.state;
  assertClose(estPos, truePos, 1.0, 'Position estimate');
  assertClose(estVel, TRUE_VEL, 0.2, 'Velocity estimate');
});

test('Innovation should be zero-mean over many measurements', () => {
  // Statistically: E[innovation] = 0 for a consistent filter
  const n = 1, m = 1;
  const F = new Float64Array([1]);
  const H = new Float64Array([1]);
  const Q = new Float64Array([0.01]);
  const R = new Float64Array([1.0]);
  const x0 = new Float64Array([0]);
  const P0 = new Float64Array([1]);

  const kf = new KalmanFilter({ n, m, F, H, Q, R, x0, P0 });
  const innovations = [];
  const TRUE = 3.0;

  for (let i = 0; i < 500; i++) {
    kf.predict();
    const z = new Float64Array([TRUE + (Math.random()-0.5) * 2]);
    const { innovation } = kf.update(z);
    innovations.push(innovation[0]);
  }

  const meanInnovation = innovations.reduce((s,v)=>s+v,0) / innovations.length;
  assertClose(meanInnovation, 0, 0.15, 'Innovation mean should be ~0');
});

test('Kalman gain should approach zero for zero measurement noise', () => {
  // If R → 0: perfect measurements, K → H⁻¹, fully trust measurements
  const n = 1, m = 1;
  const F = new Float64Array([1]);
  const H = new Float64Array([1]);
  const Q = new Float64Array([1.0]);
  const R = new Float64Array([1e-8]);  // near-zero measurement noise
  const x0 = new Float64Array([0]);
  const P0 = new Float64Array([100]);

  const kf = new KalmanFilter({ n, m, F, H, Q, R, x0, P0 });
  kf.predict();
  const { K } = kf.update(new Float64Array([5.0]));
  // K should be ≈ 1 (trust measurement almost entirely)
  assertClose(K[0], 1, 0.01, 'Kalman gain should be ~1 for zero R');
});

test('Posterior covariance should be positive semi-definite', () => {
  const n = 2, m = 2;
  const F = new Float64Array([1,0.1, 0,1]);
  const H = new Float64Array([1,0, 0,1]);
  const Q = new Float64Array([0.01,0, 0,0.01]);
  const R = new Float64Array([1,0, 0,1]);
  const x0 = new Float64Array([0,0]);
  const P0 = new Float64Array([10,0, 0,10]);

  const kf = new KalmanFilter({ n, m, F, H, Q, R, x0, P0 });

  for (let i = 0; i < 50; i++) {
    kf.predict();
    kf.update(new Float64Array([Math.random()*10, Math.random()*10]));
  }

  const P = kf.covariance;
  // PSD: diagonal elements non-negative, det ≥ 0
  assert(P[0] >= 0, 'P[0,0] ≥ 0');
  assert(P[3] >= 0, 'P[1,1] ≥ 0');
  assert(P[0]*P[3] - P[1]*P[2] >= -1e-10, 'det(P) ≥ 0 (PSD)');
});

// ─── NEES Test (Statistical Consistency) ─────────────────────────────────────

test('NEES should be χ²(n) distributed — consistency check', () => {
  // NEES = (x_true - x_est)ᵀ P⁻¹ (x_true - x_est)
  // Should be χ²(1) distributed for scalar state → E[NEES] ≈ 1
  // Run M Monte Carlo trials

  const M = 500;
  const neesValues = [];
  const TRUE_POS = 7.0;

  for (let trial = 0; trial < M; trial++) {
    const n = 1, m = 1;
    const kf = new KalmanFilter({
      n, m,
      F: new Float64Array([1]),
      H: new Float64Array([1]),
      Q: new Float64Array([0.1]),
      R: new Float64Array([1.0]),
      x0: new Float64Array([0]),
      P0: new Float64Array([100]),
    });

    for (let i = 0; i < 30; i++) {
      kf.predict();
      kf.update(new Float64Array([TRUE_POS + Math.sqrt(1.0) * (Math.random()*2-1)]));
    }

    const error = TRUE_POS - kf.state[0];
    const P = kf.covariance[0];
    neesValues.push(error * error / P);
  }

  const meanNEES = neesValues.reduce((s,v)=>s+v,0) / M;
  // For n=1 state: E[NEES] = 1.0, with ±3σ/√M interval ≈ ±0.3 for M=500
  assertClose(meanNEES, 1.0, 0.35, `Mean NEES (got ${meanNEES.toFixed(3)})`);
});

// ─── Mahalanobis Gating Tests ─────────────────────────────────────────────────

console.log('\n── Mahalanobis Gating ──');

import { mahalanobisGate } from '../src/implementation.js';

test('On-distribution measurement should pass gate', () => {
  const innovation = new Float64Array([0.5, 0.5]);
  const S = new Float64Array([1, 0, 0, 1]);  // identity covariance
  const { accepted, distance } = mahalanobisGate(innovation, S, 2, 9.488);
  assert(accepted, `d²=${distance.toFixed(2)} should pass χ²(2,0.05)=9.488 gate`);
});

test('Far-out-of-distribution measurement should be rejected', () => {
  const innovation = new Float64Array([5, 5]);  // d² = 50 >> 9.488
  const S = new Float64Array([1, 0, 0, 1]);
  const { accepted, distance } = mahalanobisGate(innovation, S, 2, 9.488);
  assert(!accepted, `d²=${distance.toFixed(2)} should be rejected`);
});

// ─── Summary ──────────────────────────────────────────────────────────────────

console.log(`\n  Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
