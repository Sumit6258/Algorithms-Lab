// FILE: shared/math/index.js
// Core math primitives — kept lean, no bloat, no big.js unless we need arbitrary precision.

'use strict';

// ─── Vector Ops ──────────────────────────────────────────────────────────────

export const Vec2 = {
  add: (a, b) => ({ x: a.x + b.x, y: a.y + b.y }),
  sub: (a, b) => ({ x: a.x - b.x, y: a.y - b.y }),
  scale: (v, s) => ({ x: v.x * s, y: v.y * s }),
  dot: (a, b) => a.x * b.x + a.y * b.y,
  len: (v) => Math.sqrt(v.x * v.x + v.y * v.y),
  norm: (v) => { const l = Vec2.len(v) || 1e-10; return { x: v.x / l, y: v.y / l }; },
  angle: (v) => Math.atan2(v.y, v.x),
  rotate: (v, θ) => ({
    x: v.x * Math.cos(θ) - v.y * Math.sin(θ),
    y: v.x * Math.sin(θ) + v.y * Math.cos(θ),
  }),
};

export const Vec3 = {
  add: (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
  sub: (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
  scale: (v, s) => [v[0]*s, v[1]*s, v[2]*s],
  dot: (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2],
  cross: (a, b) => [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ],
  len: (v) => Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]),
  norm: (v) => { const l = Vec3.len(v) || 1e-12; return [v[0]/l, v[1]/l, v[2]/l]; },
  lerp: (a, b, t) => [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t],
};

// ─── Matrix Ops ──────────────────────────────────────────────────────────────
// Column-major flat arrays, compatible with WebGL uniform upload.

export const Mat = {
  // nxn identity
  identity: (n) => Float64Array.from({ length: n*n }, (_, i) => (i % (n+1) === 0 ? 1 : 0)),

  // Matrix multiply: C = A @ B, both stored row-major for simplicity here
  mul: (A, B, n) => {
    const C = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < n; k++) {
        const aik = A[i*n + k];
        for (let j = 0; j < n; j++)
          C[i*n + j] += aik * B[k*n + j];
      }
    return C;
  },

  // Transpose in-place (square)
  transpose: (A, n) => {
    const B = new Float64Array(A.length);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        B[j*n + i] = A[i*n + j];
    return B;
  },

  // Matrix-vector product
  mulVec: (A, v, n) => {
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        r[i] += A[i*n + j] * v[j];
    return r;
  },

  // Cholesky decomposition for positive definite systems (used in Kalman)
  // Returns lower triangular L such that A = L @ L^T
  cholesky: (A, n) => {
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = A[i*n + j];
        for (let k = 0; k < j; k++) sum -= L[i*n + k] * L[j*n + k];
        L[i*n + j] = (i === j) ? Math.sqrt(Math.max(sum, 0)) : sum / L[j*n + j];
      }
    }
    return L;
  },

  // Solve lower triangular Lx = b (forward substitution)
  solveLower: (L, b, n) => {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let j = 0; j < i; j++) s -= L[i*n + j] * x[j];
      x[i] = s / L[i*n + i];
    }
    return x;
  },

  // Solve upper triangular Ux = b (back substitution)
  solveUpper: (U, b, n) => {
    const x = new Float64Array(n);
    for (let i = n-1; i >= 0; i--) {
      let s = b[i];
      for (let j = i+1; j < n; j++) s -= U[i*n + j] * x[j];
      x[i] = s / U[i*n + i];
    }
    return x;
  },

  // Add scalar * identity (regularization trick)
  addDiag: (A, λ, n) => {
    const B = new Float64Array(A);
    for (let i = 0; i < n; i++) B[i*n + i] += λ;
    return B;
  },
};

// ─── Statistics ──────────────────────────────────────────────────────────────

export const Stats = {
  mean: (arr) => arr.reduce((s, x) => s + x, 0) / arr.length,

  variance: (arr) => {
    const μ = Stats.mean(arr);
    return arr.reduce((s, x) => s + (x - μ) ** 2, 0) / arr.length;
  },

  std: (arr) => Math.sqrt(Stats.variance(arr)),

  covariance: (x, y) => {
    const μx = Stats.mean(x), μy = Stats.mean(y);
    return x.reduce((s, xi, i) => s + (xi - μx) * (y[i] - μy), 0) / x.length;
  },

  // Box-Muller — used constantly in Monte Carlo / stochastic simulations
  gaussianRandom: (μ = 0, σ = 1) => {
    const u1 = Math.random(), u2 = Math.random();
    return μ + σ * Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
  },

  // Multivariate normal sample from L (cholesky factor)
  mvnSample: (μ, L, n) => {
    const z = Float64Array.from({ length: n }, () => Stats.gaussianRandom());
    return z.map((_, i) => μ[i] + L.slice(i*n, i*n+n).reduce((s, lij, j) => s + lij * z[j], 0));
  },

  // Exponential moving average — inline to avoid repeated function call overhead
  ema: (prev, next, α) => α * next + (1 - α) * prev,

  percentile: (sorted, p) => {
    const idx = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  },
};

// ─── Numerical / Optimization ────────────────────────────────────────────────

export const Numerics = {
  // Brent's method for scalar root finding — more robust than Newton when derivative is absent
  brent: (f, a, b, tol = 1e-9, maxIter = 100) => {
    let fa = f(a), fb = f(b);
    if (fa * fb > 0) throw new Error('Root not bracketed');
    if (Math.abs(fa) < Math.abs(fb)) { [a, b] = [b, a]; [fa, fb] = [fb, fa]; }
    let c = a, fc = fa, mflag = true, s = 0, d = 0;
    for (let i = 0; i < maxIter; i++) {
      if (Math.abs(b - a) < tol) return (a + b) / 2;
      if (fa !== fc && fb !== fc) {
        s = a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) + c*fa*fb/((fc-fa)*(fc-fb));
      } else {
        s = b - fb * (b - a) / (fb - fa);
      }
      const cond = (s < (3*a+b)/4 || s > b) ||
        (mflag && Math.abs(s-b) >= Math.abs(b-c)/2) ||
        (!mflag && Math.abs(s-b) >= Math.abs(c-d)/2);
      if (cond) { s = (a+b)/2; mflag = true; } else mflag = false;
      const fs = f(s); d = c; c = b; fc = fb;
      if (fa * fs < 0) { b = s; fb = fs; } else { a = s; fa = fs; }
      if (Math.abs(fa) < Math.abs(fb)) { [a, b] = [b, a]; [fa, fb] = [fb, fa]; }
    }
    return (a + b) / 2;
  },

  // 4th-order Runge-Kutta — standard workhorse for ODE integration
  rk4: (f, y, t, dt) => {
    const k1 = f(t, y);
    const k2 = f(t + dt/2, y.map((yi, i) => yi + dt/2 * k1[i]));
    const k3 = f(t + dt/2, y.map((yi, i) => yi + dt/2 * k2[i]));
    const k4 = f(t + dt, y.map((yi, i) => yi + dt * k3[i]));
    return y.map((yi, i) => yi + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]));
  },

  // Softmax — numerically stable (subtract max before exp)
  softmax: (logits) => {
    const max = Math.max(...logits);
    const exp = logits.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sum);
  },

  // Sigmoid — clamped to avoid overflow in float32 contexts
  sigmoid: (x) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),

  // Log-sum-exp trick
  logSumExp: (arr) => {
    const max = Math.max(...arr);
    return max + Math.log(arr.reduce((s, x) => s + Math.exp(x - max), 0));
  },

  // Clamp
  clamp: (x, lo, hi) => Math.max(lo, Math.min(hi, x)),
};

// ─── Geometry / Space ────────────────────────────────────────────────────────

export const Geo = {
  // Great-circle distance (Haversine), returns meters
  haversine: (lat1, lon1, lat2, lon2) => {
    const R = 6371e3;
    const φ1 = lat1 * Math.PI/180, φ2 = lat2 * Math.PI/180;
    const dφ = (lat2 - lat1) * Math.PI/180;
    const dλ = (lon2 - lon1) * Math.PI/180;
    const a = Math.sin(dφ/2)**2 + Math.cos(φ1)*Math.cos(φ2)*Math.sin(dλ/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  },

  // Euler angle to rotation matrix (ZYX convention — aerospace standard)
  euler2rot: (roll, pitch, yaw) => {
    const cr = Math.cos(roll), sr = Math.sin(roll);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    return new Float64Array([
      cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
      sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
      -sp,   cp*sr,            cp*cr
    ]);
  },

  // Quaternion multiply (Hamilton product)
  quatMul: (q, r) => [
    q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
    q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
    q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
    q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0],
  ],

  quatNorm: (q) => { const l = Math.sqrt(q.reduce((s,x) => s+x*x, 0)); return q.map(x=>x/l); },

  // Rodrigues' rotation formula — rotate v by angle θ around axis k
  rodrigues: (v, k, θ) => {
    const cosθ = Math.cos(θ), sinθ = Math.sin(θ);
    const dot = Vec3.dot(v, k);
    const cross = Vec3.cross(k, v);
    return v.map((vi, i) => vi*cosθ + cross[i]*sinθ + k[i]*dot*(1-cosθ));
  },
};

// ─── Signal Processing ───────────────────────────────────────────────────────

export const Signal = {
  // Cooley-Tukey FFT (radix-2, in-place, real input via split-radix trick omitted for brevity)
  fft: (re, im) => {
    const n = re.length;
    if (n <= 1) return;
    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let j = 0; j < len/2; j++) {
          const uRe = re[i+j], uIm = im[i+j];
          const vRe = re[i+j+len/2]*curRe - im[i+j+len/2]*curIm;
          const vIm = re[i+j+len/2]*curIm + im[i+j+len/2]*curRe;
          re[i+j] = uRe + vRe; im[i+j] = uIm + vIm;
          re[i+j+len/2] = uRe - vRe; im[i+j+len/2] = uIm - vIm;
          [curRe, curIm] = [curRe*wRe - curIm*wIm, curRe*wIm + curIm*wRe];
        }
      }
    }
  },

  // Simple FIR low-pass with Hamming window
  firLowPass: (cutoff, numTaps) => {
    const h = new Float64Array(numTaps);
    const mid = (numTaps - 1) / 2;
    for (let i = 0; i < numTaps; i++) {
      const hamming = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (numTaps - 1));
      const sinc = i === mid ? 2 * cutoff : Math.sin(2*Math.PI*cutoff*(i-mid)) / (Math.PI*(i-mid));
      h[i] = sinc * hamming;
    }
    return h;
  },

  convolve: (signal, kernel) => {
    const n = signal.length, m = kernel.length;
    const out = new Float64Array(n + m - 1);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++)
        out[i+j] += signal[i] * kernel[j];
    return out;
  },
};

export default { Vec2, Vec3, Mat, Stats, Numerics, Geo, Signal };
