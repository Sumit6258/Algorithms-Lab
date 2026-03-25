// FILE: aerospace/missile-guidance/src/implementation.js
//
// Proportional Navigation (PN) guidance law — used in virtually all modern
// tactical missiles (AIM-120, MBDA Meteor, Rafael Python 5, etc.)
// Also covers Augmented PN (APN) and Optimal Guidance Law (OGL).
//
// PN principle: acceleration command ∝ LOS rotation rate × closing velocity
// Physics: if LOS rate → 0, missile and target are on a collision triangle.
// The missile needs zero terminal heading error — PN guarantees this with
// finite maneuver budget for non-maneuvering targets.
//
// Units: meters, seconds, radians. NED (North-East-Down) reference frame.

'use strict';

const G = 9.80665;  // m/s²

// ─── Basic PN Guidance ────────────────────────────────────────────────────────

/**
 * Pure Proportional Navigation (PN) guidance command.
 * a_cmd = N' · Vc · λ̇
 * where:
 *   N' = effective navigation ratio (typically 3-5 for PN, >4 for APN)
 *   Vc = closing velocity (positive = approaching)
 *   λ̇  = LOS (line-of-sight) rotation rate
 *
 * @param {number[3]} rM  Missile position [x,y,z]
 * @param {number[3]} vM  Missile velocity
 * @param {number[3]} rT  Target position
 * @param {number[3]} vT  Target velocity
 * @param {number}    N   Navigation constant (3-5 typical)
 * @returns {{ acmd: number[3], los: number[3], losDot: number[3], Vc: number }}
 */
export function proportionalNavigation(rM, vM, rT, vT, N = 4) {
  const R  = vec3sub(rT, rM);          // range vector (missile → target)
  const Rd = vec3sub(vT, vM);          // relative velocity
  const r_mag = vec3len(R);

  if (r_mag < 0.1) return { acmd: [0,0,0], los: R, losDot: [0,0,0], Vc: 0, timeToGo: 0 };

  const r_hat = vec3norm(R);
  const Vc    = -vec3dot(Rd, r_hat);   // closing velocity (positive = closing)
  const tgo   = r_mag / Math.max(Vc, 0.1);

  // LOS rate: λ̇ = (R × Ṙ) / |R|² — angular velocity vector of LOS
  const cross = vec3cross(R, Rd);
  const losDot = vec3scale(cross, 1 / (r_mag * r_mag));

  // PN command: a_cmd = N' · Vc · (losDot × r_hat)
  // This ensures acceleration is perpendicular to LOS
  const acmd_dir = vec3cross(losDot, r_hat);
  const acmd = vec3scale(acmd_dir, N * Vc);

  return { acmd, los: R, losDot, Vc, timeToGo: tgo, range: r_mag };
}

// ─── Augmented Proportional Navigation ───────────────────────────────────────

/**
 * Augmented PN adds target acceleration compensation.
 * a_cmd = N'·Vc·λ̇ + (N'/2)·aT_perp
 * This halves the required terminal maneuver against a constant-acceleration target.
 * Used in BVR (beyond visual range) missiles where target is maneuveringhard during endgame.
 */
export function augmentedPN(rM, vM, rT, vT, aT, N = 4) {
  const base = proportionalNavigation(rM, vM, rT, vT, N);
  const r_hat = vec3norm(base.los);

  // Target acceleration perpendicular to LOS
  const aT_parallel = vec3scale(r_hat, vec3dot(aT, r_hat));
  const aT_perp     = vec3sub(aT, aT_parallel);

  const aT_compensation = vec3scale(aT_perp, N / 2);
  const acmd_total = vec3add(base.acmd, aT_compensation);

  return { ...base, acmd: acmd_total, targetAccelCompensation: aT_compensation };
}

// ─── Optimal Guidance Law ─────────────────────────────────────────────────────

/**
 * OGL (Optimal Guidance Law) — minimizes terminal miss distance for
 * maneuvering targets with bounded acceleration.
 * Reduces to PN when target acceleration = 0.
 *
 * a_cmd = 6·r/(tgo²)·(λ - φ_f/2) + 2·ṙ/tgo + 3·aT_perp/2
 * (simplified planar version)
 *
 * This is derived from the solution to the two-point boundary value problem
 * with bounded final heading error — more formally, it's the solution to
 * the Riccati equation for the pursuit-evasion zero-sum game.
 */
export function optimalGuidanceLaw(rM, vM, rT, vT, aT = [0,0,0], N = 4) {
  const R  = vec3sub(rT, rM);
  const Rd = vec3sub(vT, vM);
  const r_mag = vec3len(R);
  const r_hat = vec3norm(R);
  const Vc    = -vec3dot(Rd, r_hat);
  const tgo   = Math.max(r_mag / Math.max(Vc, 1), 0.01);

  // Relative kinematics
  const r_perp = vec3sub(R, vec3scale(r_hat, vec3dot(R, r_hat)));  // zero for collinear
  const v_perp = vec3sub(Rd, vec3scale(r_hat, vec3dot(Rd, r_hat)));

  // OGL gain for constant-acceleration target
  const aT_perp = vec3sub(aT, vec3scale(r_hat, vec3dot(aT, r_hat)));

  const term1 = vec3scale(v_perp, 2 / tgo);
  const term2 = vec3scale(aT_perp, 1.5);

  // N=6 corresponds to optimal; lower N trades miss distance for lower peak acceleration
  const acmd = vec3add(vec3add(
    vec3scale(v_perp, N / tgo),
    term1
  ), term2);

  return { acmd, tgo, range: r_mag, Vc };
}

// ─── 6DOF Missile Dynamics ────────────────────────────────────────────────────

/**
 * Simplified 6DOF (6 degrees of freedom) missile dynamics.
 * Full 6DOF would track attitude quaternion + angular rates separately.
 * This point-mass model with first-order lag is adequate for engagement geometry.
 */
export class MissileSimulator {
  constructor(config = {}) {
    const {
      mass0     = 200,     // kg (initial, with fuel)
      massf     = 100,     // kg (burnout)
      Isp       = 240,     // specific impulse, seconds (solid rocket motor)
      burnTime  = 5,       // seconds
      Cd        = 0.3,     // drag coefficient
      area      = 0.05,    // m² reference area (≈25cm diameter)
      maxG      = 40,      // maximum lateral acceleration in G
      timeConst = 0.1,     // first-order airframe lag (seconds)
    } = config;

    this.mass0 = mass0; this.massf = massf;
    this.Isp = Isp; this.burnTime = burnTime;
    this.Cd = Cd; this.area = area;
    this.maxG = maxG; this.τ = timeConst;

    this.reset();
  }

  reset(r0 = [0,0,0], v0 = [0, 200, 0]) {
    this.position = Float64Array.from(r0);
    this.velocity = Float64Array.from(v0);
    this.accel    = new Float64Array(3);
    this.t        = 0;
    this.mass     = this.mass0;
    this.alive    = true;
    this.history  = [];
  }

  get speed() { return vec3len(Array.from(this.velocity)); }
  get altitude() { return -this.position[2]; }  // NED: Z is down

  _atmosphere(alt) {
    // ISA model up to 11km; simplified exponential above
    if (alt < 11000) {
      const T = 288.15 - 0.0065 * alt;
      const ρ = 1.225 * (T / 288.15) ** 4.256;
      return { rho: ρ, T };
    }
    return { rho: 0.3639 * Math.exp(-(alt - 11000) / 6341.6), T: 216.65 };
  }

  _thrustAndMassBurn(t) {
    if (t >= this.burnTime) return { thrust: 0, mdot: 0 };
    const mdot  = (this.mass0 - this.massf) / this.burnTime;
    const thrust = mdot * this.Isp * G;
    return { thrust, mdot };
  }

  /**
   * Integrate missile state for one timestep using RK4.
   * @param {number[3]} acmdPN  Guidance command from PN/APN/OGL
   * @param {number} dt  Timestep in seconds
   */
  step(acmdPN, dt) {
    if (!this.alive) return;

    const { t, mass } = this;
    const { rho }   = this._atmosphere(this.altitude);
    const { thrust, mdot } = this._thrustAndMassBurn(t);

    // Clamp guidance command to max-G envelope
    const acmdMag = vec3len(acmdPN);
    const acmd = acmdMag > this.maxG * G
      ? vec3scale(vec3norm(acmdPN), this.maxG * G)
      : acmdPN;

    // First-order lag on commanded acceleration (airframe response)
    const aFilt = acmd.map((ai, i) => this.accel[i] + (ai - this.accel[i]) * dt / this.τ);

    const v_hat = vec3norm(Array.from(this.velocity));
    const dynPress = 0.5 * rho * this.speed**2;
    const drag = dynPress * this.Cd * this.area;

    const ẍ = (t, [r,v]) => {
      const F_thrust = v_hat.map(vi => vi * thrust);
      const F_drag   = v_hat.map(vi => -vi * drag);
      const F_grav   = [0, 0, mass * G];  // NED: gravity is +Z (downward)
      const F_lat    = aFilt.map(ai => ai * mass);
      const F_total  = [0,1,2].map(i => F_thrust[i] + F_drag[i] + F_grav[i] + F_lat[i]);
      return F_total.map(f => f / mass);
    };

    // RK4
    const r = Array.from(this.position), v = Array.from(this.velocity);
    const k1v = ẍ(t, [r,v]);
    const k1r = v;
    const k2v = ẍ(t+dt/2, [r.map((ri,i)=>ri+k1r[i]*dt/2), v.map((vi,i)=>vi+k1v[i]*dt/2)]);
    const k2r = v.map((vi,i) => vi+k1v[i]*dt/2);
    const k3v = ẍ(t+dt/2, [r.map((ri,i)=>ri+k2r[i]*dt/2), v.map((vi,i)=>vi+k2v[i]*dt/2)]);
    const k3r = v.map((vi,i) => vi+k2v[i]*dt/2);
    const k4v = ẍ(t+dt, [r.map((ri,i)=>ri+k3r[i]*dt), v.map((vi,i)=>vi+k3v[i]*dt)]);

    for (let i=0;i<3;i++) {
      this.position[i] = r[i] + dt/6*(k1r[i]+2*k2r[i]+2*k3r[i]+v[i]+k4v[i]*dt);
      this.velocity[i] = v[i] + dt/6*(k1v[i]+2*k2v[i]+2*k3v[i]+k4v[i]);
      this.accel[i]    = aFilt[i];
    }

    this.mass = Math.max(this.massf, mass - mdot * dt);
    this.t   += dt;

    if (this.altitude < 0) { this.alive = false; }

    const state = {
      t: this.t, position: Array.from(this.position),
      velocity: Array.from(this.velocity), speed: this.speed,
      altitude: this.altitude, mass: this.mass,
      thrust, drag, acmd: Array.from(this.accel),
    };
    this.history.push(state);
    return state;
  }
}

// ─── Engagement Simulation ────────────────────────────────────────────────────

/**
 * Full engagement simulation: missile + target + guidance law.
 * Returns miss distance (meters) and full trajectory history.
 */
export function simulateEngagement(config = {}) {
  const {
    guidanceLaw = 'PN',
    N           = 4,
    dt          = 0.01,
    maxTime     = 60,
    missileInit = { r: [0,0,-1000], v: [0, 300, 0] },        // 1km altitude, 300m/s
    targetInit  = { r: [5000, 0, -3000], v: [-100, 0, 50] }, // 5km downrange, -3km alt
    targetAccel = [0, 0, 0],  // target maneuver in m/s²
  } = config;

  const missile = new MissileSimulator();
  missile.reset(missileInit.r, missileInit.v);

  const tgtPos = Float64Array.from(targetInit.r);
  const tgtVel = Float64Array.from(targetInit.v);
  const targetHistory = [];

  let closestApproach = Infinity;
  let closestTime = 0;
  const results = [];

  for (let t = 0; t < maxTime && missile.alive; t += dt) {
    // Target dynamics (constant velocity + programmed maneuver)
    for (let i = 0; i < 3; i++) {
      tgtVel[i] += targetAccel[i] * dt;
      tgtPos[i] += tgtVel[i] * dt;
    }
    targetHistory.push({ t, pos: Array.from(tgtPos), vel: Array.from(tgtVel) });

    // Compute guidance command
    let guidance;
    switch (guidanceLaw) {
      case 'APN': guidance = augmentedPN(Array.from(missile.position), Array.from(missile.velocity), Array.from(tgtPos), Array.from(tgtVel), targetAccel, N); break;
      case 'OGL': guidance = optimalGuidanceLaw(Array.from(missile.position), Array.from(missile.velocity), Array.from(tgtPos), Array.from(tgtVel), targetAccel, N); break;
      default:    guidance = proportionalNavigation(Array.from(missile.position), Array.from(missile.velocity), Array.from(tgtPos), Array.from(tgtVel), N);
    }

    if (guidance.timeToGo < dt * 2) break;  // terminal — stop

    const state = missile.step(guidance.acmd, dt);
    const range = vec3len(vec3sub(Array.from(missile.position), Array.from(tgtPos)));
    if (range < closestApproach) { closestApproach = range; closestTime = t; }

    results.push({ ...state, targetPos: Array.from(tgtPos), range, guidance });
  }

  return {
    missDistance:   closestApproach,
    impactTime:     closestTime,
    trajectory:     missile.history,
    targetHistory,
    results,
    guidance:       guidanceLaw,
    intercept:      closestApproach < 10,  // within kill radius
  };
}

// ─── Vector Utilities ─────────────────────────────────────────────────────────

function vec3len(v)     { return Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
function vec3norm(v)    { const l=vec3len(v)||1e-12; return [v[0]/l,v[1]/l,v[2]/l]; }
function vec3dot(a,b)   { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function vec3cross(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function vec3add(a,b)   { return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]; }
function vec3sub(a,b)   { return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function vec3scale(v,s) { return [v[0]*s,v[1]*s,v[2]*s]; }
