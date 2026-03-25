// FILE: cybersecurity/blockchain/src/implementation.js
//
// Bitcoin-style blockchain with SHA-256 PoW, Merkle trees, and difficulty adjustment.
// Not production Ethereum/Bitcoin — but algorithmically faithful to both.
// Uses Web Crypto API for SHA-256 (browser) with a sync fallback for Node.js.
//
// The difficulty adjustment (DAA) follows Bitcoin's: retarget every 2016 blocks
// to hit a 10-minute target. We use a simplified version with 10-block epochs
// for demo purposes — same math, faster cycle.

'use strict';

// ─── SHA-256 Wrapper ──────────────────────────────────────────────────────────

export async function sha256(data) {
  const bytes = typeof data === 'string'
    ? new TextEncoder().encode(data)
    : data;
  const buf = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(buf))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

// Synchronous hex SHA-256 using a self-contained implementation
// (needed for mining loop which can't be async in the hot path)
export function sha256Sync(str) {
  const msg = typeof str === 'string' ? str : JSON.stringify(str);

  // FIPS 180-4 constants
  const K = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
  ];

  const utf8 = unescape(encodeURIComponent(msg));
  const bytes = new Uint8Array(utf8.length);
  for (let i = 0; i < utf8.length; i++) bytes[i] = utf8.charCodeAt(i);

  // Pre-processing: padding
  const bitLen = bytes.length * 8;
  const padLen = ((bytes.length + 8) >> 6) + 1;
  const data = new Uint32Array(padLen * 16);
  for (let i = 0; i < bytes.length; i++) data[i>>2] |= bytes[i] << (24 - (i&3)*8);
  data[bytes.length >> 2] |= 0x80 << (24 - (bytes.length&3)*8);
  data[padLen*16 - 1] = bitLen;
  data[padLen*16 - 2] = Math.floor(bitLen / 0x100000000);

  // Initial hash values (first 32 bits of fractional parts of square roots of primes 2-19)
  let [h0,h1,h2,h3,h4,h5,h6,h7] = [
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
  ];

  const rotr = (x, n) => (x >>> n) | (x << (32-n));
  const add  = (...args) => args.reduce((a,b) => (a + b) >>> 0);

  for (let blk = 0; blk < padLen; blk++) {
    const W = new Uint32Array(64);
    for (let i = 0; i < 16; i++) W[i] = data[blk*16+i];
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(W[i-15],7) ^ rotr(W[i-15],18) ^ (W[i-15]>>>3);
      const s1 = rotr(W[i-2],17) ^ rotr(W[i-2],19)  ^ (W[i-2]>>>10);
      W[i] = add(W[i-16], s0, W[i-7], s1);
    }

    let [a,b,c,d,e,f,g,h] = [h0,h1,h2,h3,h4,h5,h6,h7];
    for (let i = 0; i < 64; i++) {
      const S1    = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
      const ch    = (e & f) ^ (~e & g);
      const temp1 = add(h, S1, ch, K[i], W[i]);
      const S0    = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
      const maj   = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = add(S0, maj);
      h = g; g = f; f = e; e = add(d, temp1);
      d = c; c = b; b = a; a = add(temp1, temp2);
    }
    h0 = add(h0,a); h1 = add(h1,b); h2 = add(h2,c); h3 = add(h3,d);
    h4 = add(h4,e); h5 = add(h5,f); h6 = add(h6,g); h7 = add(h7,h);
  }

  return [h0,h1,h2,h3,h4,h5,h6,h7].map(v => v.toString(16).padStart(8,'0')).join('');
}

// ─── Merkle Tree ──────────────────────────────────────────────────────────────

/**
 * Binary Merkle tree over transactions.
 * Properties:
 *   - O(log n) inclusion proofs (audit path)
 *   - Tampering any tx invalidates the root and all ancestor hashes
 *   - Odd leaf count: duplicate the last leaf (Bitcoin convention)
 */
export class MerkleTree {
  constructor(leaves) {
    if (!leaves.length) throw new Error('Empty Merkle tree');
    this.leaves  = leaves.map(tx => sha256Sync(JSON.stringify(tx)));
    this.root    = this._buildTree(this.leaves);
    this._layers = this._computeLayers(this.leaves);
  }

  _buildTree(hashes) {
    if (hashes.length === 1) return hashes[0];
    if (hashes.length % 2 !== 0) hashes = [...hashes, hashes[hashes.length-1]];
    const next = [];
    for (let i = 0; i < hashes.length; i += 2) {
      next.push(sha256Sync(hashes[i] + hashes[i+1]));
    }
    return this._buildTree(next);
  }

  _computeLayers(hashes) {
    const layers = [hashes.slice()];
    let current = hashes.slice();
    while (current.length > 1) {
      if (current.length % 2 !== 0) current = [...current, current[current.length-1]];
      const next = [];
      for (let i = 0; i < current.length; i += 2) next.push(sha256Sync(current[i]+current[i+1]));
      layers.push(next);
      current = next;
    }
    return layers;
  }

  /**
   * Generate inclusion proof for leaf at index `idx`.
   * Returns array of {hash, side} pairs — the "audit path".
   */
  getProof(idx) {
    const proof = [];
    let i = idx;
    for (const layer of this._layers.slice(0, -1)) {
      if (layer.length % 2 !== 0) layer.push(layer[layer.length-1]);
      const sibling = i % 2 === 0 ? i + 1 : i - 1;
      if (sibling < layer.length) {
        proof.push({ hash: layer[sibling], side: i % 2 === 0 ? 'right' : 'left' });
      }
      i = Math.floor(i / 2);
    }
    return proof;
  }

  /**
   * Verify a proof. Returns true if the leaf hashes up to the stored root.
   */
  verify(leafData, proof, root) {
    let hash = sha256Sync(JSON.stringify(leafData));
    for (const { hash: sibHash, side } of proof) {
      hash = side === 'right'
        ? sha256Sync(hash + sibHash)
        : sha256Sync(sibHash + hash);
    }
    return hash === root;
  }
}

// ─── Transaction ──────────────────────────────────────────────────────────────

export class Transaction {
  constructor({ from, to, amount, data = null }) {
    if (amount <= 0) throw new Error('Transaction amount must be positive');
    this.from      = from;
    this.to        = to;
    this.amount    = amount;
    this.data      = data;
    this.timestamp = Date.now();
    this.id        = sha256Sync(`${from}${to}${amount}${this.timestamp}${Math.random()}`);
  }

  toJSON() {
    return { id: this.id, from: this.from, to: this.to, amount: this.amount, timestamp: this.timestamp, data: this.data };
  }
}

// ─── Block ────────────────────────────────────────────────────────────────────

export class Block {
  constructor({ index, transactions, previousHash, difficulty, miner = 'unknown' }) {
    this.index        = index;
    this.transactions = transactions;
    this.previousHash = previousHash;
    this.difficulty   = difficulty;
    this.miner        = miner;
    this.timestamp    = Date.now();
    this.nonce        = 0;
    this.merkleRoot   = new MerkleTree(transactions.map(tx => tx.toJSON())).root;
    this.hash         = '';  // set during mining
  }

  computeHash() {
    return sha256Sync(JSON.stringify({
      index:      this.index,
      merkleRoot: this.merkleRoot,
      prevHash:   this.previousHash,
      timestamp:  this.timestamp,
      nonce:      this.nonce,
      difficulty: this.difficulty,
    }));
  }

  /**
   * Mine the block: find nonce such that hash starts with `difficulty` zeros.
   * Returns { hash, nonce, attempts, timeMs }.
   *
   * In production Bitcoin, difficulty is ~17 leading zero bits (not hex chars),
   * and mining is done at TH/s with ASICs. At difficulty=4 (16-bit), JS can
   * find a valid hash in ~65k attempts on average.
   */
  mine(onProgress = null) {
    const target = '0'.repeat(this.difficulty);
    const start  = performance.now();
    let attempts = 0;

    while (true) {
      this.hash = this.computeHash();
      attempts++;

      if (this.hash.startsWith(target)) {
        const timeMs = performance.now() - start;
        return { hash: this.hash, nonce: this.nonce, attempts, timeMs, hashRate: attempts / (timeMs / 1000) };
      }

      this.nonce++;
      if (onProgress && attempts % 1000 === 0) onProgress({ attempts, hash: this.hash });

      // Yield control every 10k iterations to prevent UI lockup
      // In a real implementation this would run in a Web Worker
      if (attempts > 5_000_000) throw new Error('Mining timeout — reduce difficulty');
    }
  }

  /**
   * Async mining using chunked iteration — keeps UI responsive.
   * Yields every 5000 hashes via setTimeout(0).
   */
  async mineAsync(onProgress = null) {
    const target = '0'.repeat(this.difficulty);
    const start = performance.now();
    let attempts = 0;

    return new Promise((resolve, reject) => {
      const chunk = () => {
        for (let i = 0; i < 5000; i++) {
          this.hash = this.computeHash();
          attempts++;
          if (this.hash.startsWith(target)) {
            const timeMs = performance.now() - start;
            resolve({ hash: this.hash, nonce: this.nonce, attempts, timeMs, hashRate: attempts / (timeMs / 1000) });
            return;
          }
          this.nonce++;
          if (attempts > 10_000_000) { reject(new Error('Mining timeout')); return; }
        }
        if (onProgress) onProgress({ attempts, hash: this.hash, nonce: this.nonce });
        setTimeout(chunk, 0);
      };
      chunk();
    });
  }

  isValid() {
    return this.hash === this.computeHash() && this.hash.startsWith('0'.repeat(this.difficulty));
  }
}

// ─── Blockchain ───────────────────────────────────────────────────────────────

export class Blockchain {
  constructor(difficulty = 3) {
    this.chain       = [this._createGenesis()];
    this.difficulty  = difficulty;
    this.pendingTxs  = [];
    this.miningReward = 6.25;  // BTC block reward (post-2020 halving)
    this.targetBlockTime = 10 * 1000;  // 10s for demo (Bitcoin: 10min)
    this.retargetInterval = 10;        // retarget every 10 blocks (Bitcoin: 2016)
  }

  _createGenesis() {
    const genesis = new Block({
      index: 0,
      transactions: [],
      previousHash: '0'.repeat(64),
      difficulty: 1,
      miner: 'genesis',
    });
    genesis.hash = sha256Sync('genesis_block_0000');
    return genesis;
  }

  get latestBlock() { return this.chain[this.chain.length - 1]; }

  addTransaction(tx) {
    if (!tx.from || !tx.to) throw new Error('Invalid transaction');
    if (tx.amount <= 0) throw new Error('Non-positive amount');
    // In production: verify digital signature here
    this.pendingTxs.push(tx);
    return this.pendingTxs.length;
  }

  /**
   * Mine pending transactions into a new block.
   * Includes a coinbase transaction (miner reward) — no mining without reward.
   */
  async minePendingTransactions(minerAddress) {
    const coinbase = new Transaction({
      from: 'COINBASE',
      to: minerAddress,
      amount: this.miningReward,
      data: { type: 'coinbase', blockHeight: this.chain.length }
    });

    const txs = [coinbase, ...this.pendingTxs.slice(0, 2000)];  // ~2000 tx/block limit (simplified; Bitcoin uses byte weight)

    const block = new Block({
      index:        this.chain.length,
      transactions: txs,
      previousHash: this.latestBlock.hash,
      difficulty:   this.difficulty,
      miner:        minerAddress,
    });

    const result = await block.mineAsync();
    this.chain.push(block);
    this.pendingTxs = this.pendingTxs.slice(txs.length - 1);  // remove mined txs

    // Difficulty adjustment
    if (this.chain.length % this.retargetInterval === 0) {
      this._adjustDifficulty();
    }

    return { block, ...result };
  }

  /**
   * Difficulty Adjustment Algorithm (DAA) — Bitcoin-style.
   * Computes actual time for last epoch vs target, scales difficulty accordingly.
   * Bitcoin caps adjustment at 4× in either direction to prevent manipulation.
   */
  _adjustDifficulty() {
    const epochStart = this.chain[this.chain.length - this.retargetInterval];
    const epochEnd   = this.latestBlock;
    const actual     = epochEnd.timestamp - epochStart.timestamp;
    const target     = this.targetBlockTime * this.retargetInterval;

    const ratio = Math.max(0.25, Math.min(4, target / actual));
    const newDifficulty = Math.max(1, Math.round(this.difficulty * ratio));

    if (newDifficulty !== this.difficulty) {
      this.difficulty = newDifficulty;
    }

    return { actual, target, ratio, newDifficulty };
  }

  /**
   * Full chain validation: hash integrity + previous hash linkage + PoW.
   * O(n) where n = chain length.
   */
  isValid() {
    for (let i = 1; i < this.chain.length; i++) {
      const cur  = this.chain[i];
      const prev = this.chain[i-1];

      if (!cur.isValid())                        return { valid: false, block: i, reason: 'invalid_hash' };
      if (cur.previousHash !== prev.hash)         return { valid: false, block: i, reason: 'broken_chain' };
      if (cur.index !== prev.index + 1)           return { valid: false, block: i, reason: 'invalid_index' };
      // Merkle root check (ensures tx set hasn't been modified post-mining)
      const recomputedRoot = new MerkleTree(cur.transactions.map(tx => tx.toJSON())).root;
      if (recomputedRoot !== cur.merkleRoot)      return { valid: false, block: i, reason: 'merkle_mismatch' };
    }
    return { valid: true };
  }

  /**
   * Get UTXO balance for address (simplified: no script validation).
   * Production Bitcoin uses a full UTXO set (currently ~5GB, stored in LevelDB).
   */
  getBalance(address) {
    let balance = 0;
    for (const block of this.chain) {
      for (const tx of block.transactions) {
        if (tx.to === address)   balance += tx.amount;
        if (tx.from === address) balance -= tx.amount;
      }
    }
    return Math.max(0, balance);
  }

  getStats() {
    return {
      height:         this.chain.length - 1,
      difficulty:     this.difficulty,
      pendingTxs:     this.pendingTxs.length,
      totalTxs:       this.chain.reduce((s, b) => s + b.transactions.length, 0),
      hashRate:       `~${(2**this.difficulty/10).toFixed(0)} H/s (estimated)`,
      totalMined:     this.chain.reduce((s, b) =>
        s + b.transactions.filter(tx => tx.from === 'COINBASE').reduce((r, tx) => r + tx.amount, 0), 0),
    };
  }
}
