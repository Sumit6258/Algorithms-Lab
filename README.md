# ALGORITHMS LAB

**Production-grade algorithm implementations with 3D WebGL visualizations.**

A monorepo spanning aerospace, AI, finance, cybersecurity, and social media systems — built to the standard of real engineering teams, not textbook examples.

---

## Structure

```
algorithms-lab/
├── aerospace/
│   ├── kalman-filter/          EKF + UKF sensor fusion
│   ├── orbital-mechanics/      Two-body + J2 perturbation + Lambert
│   ├── missile-guidance/       PN + APN + OGL guidance laws
│   ├── rocket-trajectory/      Pontryagin max principle optimization
│   └── radar-tracking/         Multi-target tracking (MHT)
│
├── artificial-intelligence/
│   ├── transformer/            Multi-head attention + RoPE + Flash Attention
│   ├── gradient-descent/       Adam, RMSProp, Lion, Adafactor
│   ├── reinforcement-learning/ Q-learning + PPO + MCTS
│   └── clustering/             K-Means++ + DBSCAN + HDBSCAN
│
├── finance/
│   ├── monte-carlo/            GBM + Heston + Sobol QMC
│   ├── portfolio-optimization/ Markowitz + Black-Litterman + Risk Parity
│   ├── hft-simulator/          LOB simulation + market making
│   └── fraud-detection/        Isolation Forest + autoencoder
│
├── cybersecurity/
│   ├── rsa-encryption/         RSA-OAEP + PSS + Miller-Rabin
│   ├── blockchain/             SHA-256 PoW + Merkle + DAA
│   ├── intrusion-detection/    Statistical anomaly detection
│   └── hashing/                SHA-256 internals + birthday paradox
│
├── social-media/
│   ├── feed-ranking/           Hybrid scoring + MMR diversity
│   ├── viral-spread/           SIR model + scale-free networks
│   ├── recommendation/         Collaborative filtering + ALS
│   └── ad-targeting/           Thompson Sampling + UCB bandits
│
├── shared/
│   └── math/                   Vec3, Mat, Stats, Numerics, FFT, RK4
│
└── visualization/
    └── index.html              WebGL/Three.js interactive portal
```

---

## Running the Visualization

No build step required — open `visualization/index.html` in a browser.

```bash
# Serve locally (avoids CORS on ES module imports)
python3 -m http.server 8080
# then open http://localhost:8080/visualization/
```

Controls:
- **Orbit**: click + drag
- **Zoom**: scroll wheel
- **Pan**: right-click + drag
- **Sliders**: adjust algorithm parameters in real-time

---

## Algorithm Notes

### Aerospace

**Kalman Filter** (`aerospace/kalman-filter/`)
Three variants: standard KF (linear), EKF (linearized), UKF (sigma-point).
The UKF is the right choice when EKF diverges — sigma points propagate the covariance through nonlinearities exactly to 3rd order. The tradeoff: 2n+1 model evaluations per step vs 1 for EKF.

**Orbital Mechanics** (`aerospace/orbital-mechanics/`)
J2 oblateness is the dominant perturbation for LEO — produces ~7°/day RAAN precession at ISS altitude. Ignoring it causes ~1 km/day position error. Lambert's problem solver uses universal variables (Battin formulation) — handles any transfer angle including 180° with care.

**Missile Guidance** (`aerospace/missile-guidance/`)
Proportional Navigation is provably optimal for non-maneuvering targets (zero miss distance with finite maneuver budget). APN adds target acceleration compensation — halves the terminal G-load requirement against a maneuvering target.

### Finance

**Monte Carlo** (`finance/monte-carlo/`)
Key insight: CVaR (Expected Shortfall) is a coherent risk measure; VaR is not (fails subadditivity). Basel IV mandates CVaR at 97.5%. Heston model captures volatility clustering and the vol smile — GBM cannot.

**Portfolio Optimization** (`finance/portfolio-optimization/`)
Markowitz frontier is computed analytically via Lagrangian (closed-form, O(n³) for matrix inverse). Black-Litterman addresses the estimation error problem in mean-variance: return forecasts are notoriously noisy, and MPT amplifies this noise into extreme corner solutions.

### Cybersecurity

**RSA** (`cybersecurity/rsa-encryption/`)
OAEP padding is mandatory — PKCS#1 v1.5 is vulnerable to Bleichenbacher's adaptive chosen-ciphertext attack. CRT optimization in private key operations gives 4× speedup by computing mod p and mod q separately (Garner's formula). In practice, use ECDH/ECDSA — RSA-2048 is equivalent security to ECDSA-224, with much larger keys.

**Blockchain** (`cybersecurity/blockchain/`)
PoW difficulty adjustment follows Bitcoin's DAA: target 10-min block time, retarget every 2016 blocks, cap at 4× change per epoch. Merkle tree enables O(log n) transaction inclusion proofs — critical for SPV (simplified payment verification) clients.

---

## Engineering Philosophy

Every algorithm here reflects real production constraints:

- **Numerical stability first**: Joseph form for KF, Cholesky with jitter, log-sum-exp trick for softmax
- **Performance awareness**: complexity annotations are honest, not aspirational
- **Failure modes documented**: "this breaks when..." is as important as "this works when..."
- **Real trade-offs**: UKF vs EKF, CVaR vs VaR, APN vs PN — not just the "best" option

---

## Dependencies

Runtime (visualization only):
- `three.js r128` — CDN, no install needed

Development (algorithm implementations):
- None — pure JS/ES modules, runs in any modern browser or Node.js 18+

---

## Testing

```bash
# Run Kalman filter tests (Node.js, requires --experimental-vm-modules for ESM)
node --experimental-vm-modules aerospace/kalman-filter/tests/test_cases.js

# Run Monte Carlo tests
node finance/monte-carlo/tests/test_cases.js
```

---

*Built to the standard: code that could ship, docs that don't lie, visualizations that reveal rather than decorate.*
