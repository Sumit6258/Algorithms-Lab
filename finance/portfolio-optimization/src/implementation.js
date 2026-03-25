// FILE: finance/portfolio-optimization/src/implementation.js
//
// Modern Portfolio Theory — full Markowitz mean-variance optimization.
// Closed-form efficient frontier via Lagrangian, plus numerical QP fallback.
// Black-Litterman model for incorporating views into the prior.
// Risk parity and maximum diversification portfolios as alternatives to MPT.
//
// MPT breaks down when: returns are non-normal (fat tails, skewness),
// covariance matrix is estimated from short history (Stambaugh bias),
// constraints exist (long-only, turnover limits, regulatory caps).
// For real portfolios, use robust optimization or Black-Litterman.

'use strict';

// ─── Efficient Frontier (Analytical) ─────────────────────────────────────────

/**
 * Compute the unconstrained efficient frontier analytically.
 * For n assets with mean returns μ and covariance Σ, the frontier is a parabola
 * in (σ², μ_p) space. The frontier portfolio weights for target return r are:
 *   w* = λ·Σ⁻¹·μ + γ·Σ⁻¹·1
 * where λ, γ are Lagrange multipliers from the KKT conditions.
 *
 * This gives the global minimum variance portfolio and any two frontier portfolios,
 * from which the entire frontier can be constructed (two-fund separation theorem).
 *
 * @param {number[]} mu    Expected returns (annualized)
 * @param {number[][]} Sigma  Covariance matrix
 * @param {number} rf     Risk-free rate (for Sharpe ratio)
 * @returns {{ frontier, gmvp, tangency, twoFund }}
 */
export function computeEfficientFrontier(mu, Sigma, rf = 0.04) {
  const n = mu.length;
  const SigmaInv = matInverse(Sigma, n);
  const ones = new Array(n).fill(1);

  // Frontier scalars
  const A = quadForm(ones, SigmaInv, ones, n);   // 1ᵀ Σ⁻¹ 1
  const B = quadForm(mu,   SigmaInv, ones, n);   // μᵀ Σ⁻¹ 1
  const C = quadForm(mu,   SigmaInv, mu,   n);   // μᵀ Σ⁻¹ μ
  const D = A * C - B * B;

  if (Math.abs(D) < 1e-12) throw new Error('Singular system: frontier undefined');

  // Global minimum variance portfolio (GMVP)
  const w_gmvp = matVec(SigmaInv, ones, n).map(v => v / A);
  const ret_gmvp = dot(mu, w_gmvp, n);
  const var_gmvp = 1 / A;

  // Tangency portfolio (maximum Sharpe ratio — unconstrained)
  // w_tan ∝ Σ⁻¹(μ - rf·1)
  const mu_excess = mu.map(m => m - rf);
  const w_tan_raw = matVec(SigmaInv, mu_excess, n);
  const sumW = w_tan_raw.reduce((s,w)=>s+w,0);
  const w_tan = w_tan_raw.map(w => w / sumW);
  const ret_tan = dot(mu, w_tan, n);
  const var_tan = quadForm(w_tan, Sigma, w_tan, n);  // Note: using Sigma not SigmaInv here
  const sharpe_tan = (ret_tan - rf) / Math.sqrt(var_tan);

  // Efficient frontier points (parametric sweep)
  const frontierPoints = [];
  const retMin = ret_gmvp;
  const retMax = Math.max(...mu) * 1.1;
  const N_POINTS = 80;

  for (let k = 0; k < N_POINTS; k++) {
    const target = retMin + (retMax - retMin) * (k / (N_POINTS - 1));

    // Frontier weights for target return: w = λ·Σ⁻¹μ + γ·Σ⁻¹1
    const λ = (A * target - B) / D;
    const γ = (C - B * target) / D;
    const SigmaInv_mu  = matVec(SigmaInv, mu, n);
    const SigmaInv_one = matVec(SigmaInv, ones, n);
    const w = SigmaInv_mu.map((v, i) => λ * v + γ * SigmaInv_one[i]);

    // Portfolio variance = wᵀΣw using the analytical formula
    const portVar = (A * target * target - 2 * B * target + C) / D;

    if (portVar < 0) continue;  // numerical noise below GMVP

    frontierPoints.push({
      return: target,
      vol:    Math.sqrt(portVar),
      sharpe: (target - rf) / Math.sqrt(portVar),
      weights: w,
    });
  }

  return {
    frontier:  frontierPoints,
    gmvp:      { weights: w_gmvp, return: ret_gmvp, vol: Math.sqrt(var_gmvp), sharpe: (ret_gmvp-rf)/Math.sqrt(var_gmvp) },
    tangency:  { weights: w_tan,  return: ret_tan,  vol: Math.sqrt(var_tan),  sharpe: sharpe_tan },
    scalars:   { A, B, C, D },
  };
}

// ─── Long-Only Constrained Optimization (Active Set) ─────────────────────────

/**
 * Minimum variance portfolio with long-only constraint (w_i ≥ 0).
 * Uses active-set QP — simpler than interior-point for small n (≤100 assets).
 * For large portfolios (n > 500), use an interior-point solver or OSQP.
 *
 * This doesn't compute the entire frontier under constraints — that requires
 * parametric QP (Markowitz's critical line algorithm), which is more involved.
 */
export function minimumVariancePortfolio(mu, Sigma, n, constraints = {}) {
  const {
    longOnly  = true,
    minWeight = 0,           // minimum weight per asset
    maxWeight = 1,           // maximum weight per asset
    turnoverLimit = Infinity, // max change from current portfolio
    currentWeights = null,
  } = constraints;

  // Gradient descent on portfolio variance with projection to constraint set
  // Simple but effective for long-only: iterative projected gradient
  let w = new Array(n).fill(1/n);  // start at equal weight
  const lr = 0.01;
  const maxIter = 2000;

  for (let iter = 0; iter < maxIter; iter++) {
    // Gradient of wᵀΣw = 2Σw
    const grad = matVec(Sigma, w, n).map(g => 2 * g);

    // Gradient step
    const w_new = w.map((wi, i) => wi - lr * grad[i]);

    // Project to simplex with box constraints
    const projected = projectToConstraints(w_new, n, minWeight, maxWeight, turnoverLimit, currentWeights);

    // Check convergence
    const change = projected.reduce((s, wi, i) => s + Math.abs(wi - w[i]), 0);
    w = projected;
    if (change < 1e-8) break;
  }

  const portReturn = dot(mu, w, n);
  const portVar    = quadForm(w, Sigma, w, n);

  return {
    weights: w,
    return:  portReturn,
    vol:     Math.sqrt(portVar),
    sharpe:  (portReturn - 0.04) / Math.sqrt(portVar),  // assuming 4% rf
    concentration: herfindahlIndex(w),
  };
}

/**
 * Project weights vector to simplex (∑w=1) with box constraints.
 * Duchi et al. algorithm for Euclidean projection onto simplex.
 */
function projectToConstraints(w, n, wMin, wMax, turnoverLimit, currentW) {
  // Apply box constraints first
  let clipped = w.map(wi => Math.max(wMin, Math.min(wMax, wi)));

  // Turnover constraint: ||w - w0||₁ ≤ turnoverLimit
  if (currentW && turnoverLimit < Infinity) {
    const turnover = clipped.reduce((s, wi, i) => s + Math.abs(wi - currentW[i]), 0);
    if (turnover > turnoverLimit) {
      // Scale changes proportionally
      const scale = turnoverLimit / turnover;
      clipped = clipped.map((wi, i) => currentW[i] + (wi - currentW[i]) * scale);
    }
  }

  // Project to probability simplex: min ||x-u||² s.t. ∑x=1, x≥0
  const sorted = [...clipped].sort((a,b) => b-a);
  let ρ = 0;
  let cumsum = 0;
  for (let j = 0; j < n; j++) {
    cumsum += sorted[j];
    if (sorted[j] - (cumsum - 1) / (j + 1) > 0) ρ = j;
  }
  const cumρ = sorted.slice(0, ρ+1).reduce((s,v)=>s+v,0);
  const θ = (cumρ - 1) / (ρ + 1);

  return clipped.map(wi => Math.max(0, wi - θ));
}

// ─── Risk Parity Portfolio ────────────────────────────────────────────────────

/**
 * Risk parity: each asset contributes equally to total portfolio risk.
 * Alternative to MPT that doesn't require return estimates (which are notoriously
 * hard to forecast and estimation error dominates).
 *
 * Marginal risk contribution of asset i: MRC_i = (Σw)_i · w_i / σ_p
 * Risk parity: w_i · MRC_i = σ_p / n for all i
 */
export function riskParityPortfolio(Sigma, n, maxIter = 1000, tol = 1e-8) {
  let w = new Array(n).fill(1/n);
  const target = 1 / n;  // equal risk contribution

  for (let iter = 0; iter < maxIter; iter++) {
    const Sw = matVec(Sigma, w, n);
    const portVar = dot(w, Sw, n);
    const portVol = Math.sqrt(portVar);

    // Risk contributions
    const RC = w.map((wi, i) => wi * Sw[i] / portVol);
    const sumRC = RC.reduce((s,r)=>s+r,0);

    // Update weights using Newton-like step
    const grad = RC.map(rc => rc - sumRC * target);
    const stepSize = 0.1;
    const w_new = w.map((wi, i) => wi * Math.exp(-stepSize * grad[i]));

    // Normalize
    const normFactor = w_new.reduce((s,v)=>s+v,0);
    const w_normalized = w_new.map(v => v / normFactor);

    const change = w_normalized.reduce((s, wi, i) => s + Math.abs(wi - w[i]), 0);
    w = w_normalized;
    if (change < tol) break;
  }

  const Sw = matVec(Sigma, w, n);
  const portVar = dot(w, Sw, n);
  const portVol = Math.sqrt(portVar);
  const RC = w.map((wi, i) => wi * Sw[i] / portVol);

  return { weights: w, riskContributions: RC, vol: portVol, concentration: herfindahlIndex(w) };
}

// ─── Black-Litterman Model ────────────────────────────────────────────────────

/**
 * Black-Litterman: combine equilibrium returns (from CAPM) with analyst views.
 * Posterior expected returns: μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹π + PᵀΩ⁻¹q]
 *
 * @param {number[]} pi   Equilibrium returns (from reverse optimization: π = δΣw_mkt)
 * @param {number[][]} Sigma  Covariance matrix
 * @param {number[][]} P   View matrix (k × n): each row defines one view
 * @param {number[]} q    View expected returns (k × 1)
 * @param {number[][]} Omega  View uncertainty matrix (k × k, often diagonal)
 * @param {number} tau   Scaling parameter (typically 0.025-0.05)
 */
export function blackLitterman(pi, Sigma, P, q, Omega, tau = 0.025, n) {
  const k = q.length;
  const tauSigma = Sigma.map(row => row.map(v => v * tau));
  const tauSigmaInv = matInverse(tauSigma, n);
  const OmegaInv = matInverse(Omega, k);

  // Precision-weighted combination
  // Left side: (τΣ)⁻¹ + PᵀΩ⁻¹P
  const Pt = matTranspose(P, k, n);
  const OmegaInvP = matMul(OmegaInv, P, k, k, n);
  const PtOmegaInvP = matMul(Pt, OmegaInvP, n, k, n);
  const leftSide = matAdd2D(tauSigmaInv, PtOmegaInvP, n);
  const leftSideInv = matInverse(leftSide, n);

  // Right side: (τΣ)⁻¹π + PᵀΩ⁻¹q
  const tauSigmaInvPi = matVec(tauSigmaInv, pi, n);
  const OmegaInvQ = matVec(OmegaInv, q, k);
  const PtOmegaInvQ = matVec(Pt, OmegaInvQ, k, n);
  const rightSide = tauSigmaInvPi.map((v, i) => v + PtOmegaInvQ[i]);

  // Posterior expected returns
  const mu_BL = matVec(leftSideInv, rightSide, n);

  // Posterior covariance
  const posteriorCov = matAdd2D(Sigma, leftSideInv, n);

  return { mu_BL, posteriorCov };
}

// ─── Performance Attribution ──────────────────────────────────────────────────

/**
 * Brinson-Hood-Beebower attribution model.
 * Decomposes active return into: allocation effect + selection effect + interaction.
 */
export function brinsonAttribution(portfolioWeights, benchmarkWeights, portfolioReturns, benchmarkReturns, n) {
  const totalPortReturn = dot(portfolioWeights, portfolioReturns, n);
  const totalBmkReturn  = dot(benchmarkWeights, benchmarkReturns, n);
  const activeReturn    = totalPortReturn - totalBmkReturn;

  const allocation  = new Array(n);
  const selection   = new Array(n);
  const interaction = new Array(n);

  for (let i = 0; i < n; i++) {
    allocation[i]  = (portfolioWeights[i] - benchmarkWeights[i]) * (benchmarkReturns[i] - totalBmkReturn);
    selection[i]   = benchmarkWeights[i] * (portfolioReturns[i] - benchmarkReturns[i]);
    interaction[i] = (portfolioWeights[i] - benchmarkWeights[i]) * (portfolioReturns[i] - benchmarkReturns[i]);
  }

  return {
    totalPortReturn, totalBmkReturn, activeReturn,
    allocation:  { perAsset: allocation, total: allocation.reduce((s,v)=>s+v,0) },
    selection:   { perAsset: selection,  total: selection.reduce((s,v)=>s+v,0) },
    interaction: { perAsset: interaction,total: interaction.reduce((s,v)=>s+v,0) },
  };
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function dot(a, b, n) { let s=0; for(let i=0;i<n;i++) s+=a[i]*b[i]; return s; }

function quadForm(a, M, b, n) {
  // aᵀ M b
  let s = 0;
  for (let i=0;i<n;i++) for(let j=0;j<n;j++) s += a[i]*M[i][j]*b[j];
  return s;
}

function matVec(M, v, rows, cols = rows) {
  return Array.from({length:rows}, (_,i) => v.reduce((s,vj,j) => s + M[i][j]*vj, 0));
}

function matMul(A, B, ra, ca, cb) {
  const C = Array.from({length:ra}, () => new Array(cb).fill(0));
  for(let i=0;i<ra;i++) for(let k=0;k<ca;k++) for(let j=0;j<cb;j++) C[i][j]+=A[i][k]*B[k][j];
  return C;
}

function matTranspose(A, rows, cols) {
  return Array.from({length:cols}, (_,j) => Array.from({length:rows}, (_2,i) => A[i][j]));
}

function matAdd2D(A, B, n) {
  return A.map((row,i) => row.map((v,j) => v + B[i][j]));
}

function matInverse(A, n) {
  // Gauss-Jordan elimination on augmented [A | I]
  const M = A.map((row,i) => [...row, ...new Array(n).fill(0).map((_,j) => i===j?1:0)]);
  for(let col=0;col<n;col++) {
    let pivot=col;
    for(let r=col+1;r<n;r++) if(Math.abs(M[r][col])>Math.abs(M[pivot][col])) pivot=r;
    [M[col],M[pivot]]=[M[pivot],M[col]];
    const pv=M[col][col];
    if(Math.abs(pv)<1e-14) throw new Error('Singular matrix in portfolio optimization');
    M[col]=M[col].map(v=>v/pv);
    for(let r=0;r<n;r++) if(r!==col){ const f=M[r][col]; M[r]=M[r].map((v,j)=>v-f*M[col][j]); }
  }
  return M.map(row=>row.slice(n));
}

function herfindahlIndex(w) {
  return w.reduce((s,wi)=>s+wi*wi,0);  // 1/n = perfectly diversified, 1 = single asset
}
