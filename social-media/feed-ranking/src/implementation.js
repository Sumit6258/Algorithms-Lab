// FILE: social-media/feed-ranking/src/implementation.js
//
// Hybrid feed ranking: engagement signals × recency decay × user affinity × diversity.
// Loosely modeled on the architecture described in Meta's "Generalized Additive Models"
// and Twitter/X's open-sourced recommender (heavy-ranker layer).
//
// The key insight: pure engagement optimization leads to outrage amplification.
// Real systems mix engagement with health signals (misinformation probability,
// user sentiment after exposure, scroll velocity — fast scroll = low quality signal).

'use strict';

// ─── Engagement Score Components ─────────────────────────────────────────────

const ENGAGEMENT_WEIGHTS = {
  // Weights reflect marginal cost to producer and value to consumer
  share:         98,   // highest-intent action
  comment:       70,   // significant time investment
  long_click:    50,   // >30s dwell
  reaction:      20,   // emotional but low-effort
  click:         10,
  hover:          3,   // weak signal, easily gamed
  impression:     1,
};

const HEALTH_WEIGHTS = {
  misinformation_score:    -200,  // hard penalty — not a soft trade-off
  user_regret_probability:  -80,  // predicted "I don't want to see this"
  hate_speech_probability:  -500,
  clickbait_title_score:    -15,  // title vs body sentiment divergence
};

// ─── Post Scoring Pipeline ────────────────────────────────────────────────────

/**
 * Main ranking function — applies in the "heavy ranker" stage after lightweight
 * candidate generation has already filtered 1M → 1000 candidates.
 *
 * @param {Post[]} candidates  ~500-2000 candidate posts
 * @param {UserProfile} user
 * @param {RankingContext} ctx  { timestamp, feedType, sessionLength }
 * @returns {ScoredPost[]}  sorted descending by final_score
 */
export function rankFeed(candidates, user, ctx) {
  const { timestamp, feedType = 'home', sessionLength = 0 } = ctx;

  const scored = candidates.map(post => {
    // ── 1. Engagement prediction (from ML model output) ──────────────────────
    const engagementScore = computeEngagementScore(post.predictedEngagement);

    // ── 2. Recency decay ──────────────────────────────────────────────────────
    const ageHours = (timestamp - post.createdAt) / 3_600_000;
    const recencyScore = recencyDecay(ageHours, feedType);

    // ── 3. User-author affinity ───────────────────────────────────────────────
    const affinityScore = computeAffinity(user, post.authorId, post.authorMeta);

    // ── 4. Content relevance ──────────────────────────────────────────────────
    const relevanceScore = computeRelevance(user.interests, post.topics, post.embedding, user.embedding);

    // ── 5. Health/safety penalties ────────────────────────────────────────────
    const healthPenalty = computeHealthPenalty(post.safetySignals);

    // ── 6. Session fatigue (avoid showing same topic repeatedly) ──────────────
    const fatiguePenalty = computeFatigue(post.topics, ctx.recentlyShownTopics, sessionLength);

    // ── 7. Social proof ───────────────────────────────────────────────────────
    const socialProof = computeSocialProof(post.stats, user.socialGraph, post.networkEngagement);

    // ── 8. Format multiplier ──────────────────────────────────────────────────
    const formatMult = FORMAT_MULTIPLIERS[post.format] ?? 1.0;

    // Raw score (pre-diversity)
    const rawScore = (
      engagementScore * 1.0 +
      recencyScore    * 0.3 +
      affinityScore   * 0.5 +
      relevanceScore  * 0.4 +
      socialProof     * 0.2
    ) * formatMult + healthPenalty - fatiguePenalty;

    return {
      post,
      scores: { engagementScore, recencyScore, affinityScore, relevanceScore, healthPenalty, rawScore },
      finalScore: rawScore,
    };
  });

  // ── 9. Diversity enforcement (MMR: Maximal Marginal Relevance) ──────────────
  const diversified = maximalMarginalRelevance(scored, user, {
    λ: 0.7,  // 0 = pure diversity, 1 = pure relevance
    targetCount: Math.min(candidates.length, 200),
  });

  // ── 10. Boost injection (ads, promoted, must-sees) ────────────────────────
  return injectBoostedContent(diversified, ctx.boostedPosts ?? []);
}

// ─── Engagement Score ─────────────────────────────────────────────────────────

function computeEngagementScore(predicted) {
  let score = 0;
  for (const [action, weight] of Object.entries(ENGAGEMENT_WEIGHTS)) {
    score += (predicted[action] ?? 0) * weight;
  }
  return score;
}

// ─── Recency Decay ────────────────────────────────────────────────────────────

/**
 * Different feed types want different recency curves.
 * 'trending': fast decay (news ages in hours)
 * 'home':     moderate decay (friends post less frequently)
 * 'explore':  slow decay (content quality > freshness)
 */
function recencyDecay(ageHours, feedType) {
  const halfLives = { trending: 2, home: 12, explore: 48, archive: Infinity };
  const τ = halfLives[feedType] ?? 12;
  if (τ === Infinity) return 50;  // static feed
  // Exponential decay with floor (some content stays relevant longer)
  return 100 * Math.exp(-Math.log(2) * ageHours / τ) + 10;
}

// ─── User-Author Affinity ─────────────────────────────────────────────────────

/**
 * Computes relationship strength between user and post author.
 * Decay: older interactions matter less (3-month half-life for passive signals).
 */
function computeAffinity(user, authorId, authorMeta) {
  const relationship = user.relationships?.[authorId];
  if (!relationship) {
    // Cold start: use follower count as rough proxy (with heavy log dampening)
    const followerScore = authorMeta?.followerCount
      ? 10 * Math.log10(1 + authorMeta.followerCount)
      : 0;
    return followerScore;
  }

  const now = Date.now();
  let affinity = 0;

  // Direct interaction signals (time-decayed)
  const INTERACTION_WEIGHTS = {
    reply: 100, dm: 150, tag: 80, like: 20, click_profile: 30, follow: 200
  };
  for (const { type, timestamp, weight: customWeight } of (relationship.interactions ?? [])) {
    const ageWeeks = (now - timestamp) / (7 * 86400 * 1000);
    const decay = Math.exp(-0.2 * ageWeeks);  // ~3.5-week half-life
    affinity += (customWeight ?? INTERACTION_WEIGHTS[type] ?? 10) * decay;
  }

  // Passive signals
  if (relationship.mutualFollows) affinity += 50;
  if (relationship.inCloseFriends) affinity += 200;
  if (relationship.blocked) return -Infinity;
  if (relationship.muted) return -100;

  return Math.min(affinity, 500);  // cap to prevent domination by super-fans
}

// ─── Content Relevance ────────────────────────────────────────────────────────

/**
 * Two-component relevance:
 *   1. Topic match (categorical): user interest tags ∩ post topics
 *   2. Embedding similarity (semantic): cosine similarity in latent space
 *
 * The embedding captures concepts that topic tags miss — e.g., a crypto post
 * tagged "finance" but embedded near "blockchain" tokens will still score high
 * for a user with a strong blockchain interest vector.
 */
function computeRelevance(userInterests, postTopics, postEmbedding, userEmbedding) {
  // Categorical component
  let topicScore = 0;
  for (const topic of (postTopics ?? [])) {
    const interest = userInterests?.[topic];
    if (interest !== undefined) topicScore += interest;  // interest in [-1, 5]
  }
  topicScore = Math.tanh(topicScore / 3) * 100;  // normalize to [-100, 100]

  // Semantic embedding component (cosine similarity)
  let embeddingScore = 0;
  if (postEmbedding && userEmbedding && postEmbedding.length === userEmbedding.length) {
    const cos = cosineSimilarity(postEmbedding, userEmbedding);
    embeddingScore = cos * 80;  // map [-1,1] → [-80,80]
  }

  return topicScore * 0.4 + embeddingScore * 0.6;
}

// ─── Health Penalties ─────────────────────────────────────────────────────────

function computeHealthPenalty(signals = {}) {
  let penalty = 0;
  for (const [signal, weight] of Object.entries(HEALTH_WEIGHTS)) {
    if (signals[signal] !== undefined) {
      penalty += signals[signal] * weight;
    }
  }
  return penalty;  // negative = penalty when added to score
}

// ─── Session Fatigue ──────────────────────────────────────────────────────────

function computeFatigue(postTopics, recentlyShownTopics = {}, sessionLength) {
  if (!postTopics?.length) return 0;

  let overlap = 0;
  for (const topic of postTopics) {
    const frequency = recentlyShownTopics[topic] ?? 0;
    // Diminishing returns: first repeat is annoying, further repeats are exponentially worse
    overlap += 1 - Math.exp(-0.5 * frequency);
  }

  // Session fatigue amplifies over time (user gets bored faster)
  const sessionFatigueMult = 1 + sessionLength * 0.02;  // 2% per minute
  return (overlap / postTopics.length) * 40 * sessionFatigueMult;
}

// ─── Social Proof ─────────────────────────────────────────────────────────────

function computeSocialProof(stats, socialGraph, networkEngagement) {
  // Connections who engaged (strongest signal)
  const friendEngagements = networkEngagement?.friends ?? [];
  const inNetworkScore = friendEngagements.length > 0
    ? 30 + 10 * Math.log(1 + friendEngagements.length)
    : 0;

  // Viral momentum (rate of engagement growth, not absolute count)
  const momentumScore = stats?.engagementVelocity
    ? 20 * Math.tanh(stats.engagementVelocity / 100)
    : 0;

  return inNetworkScore + momentumScore;
}

// ─── Maximal Marginal Relevance ───────────────────────────────────────────────

/**
 * MMR selects items by greedily maximizing:
 *   MMR_i = λ · Relevance(i) - (1-λ) · max_{j∈selected} Similarity(i,j)
 *
 * This is O(n²) in the worst case — acceptable for 200 candidates,
 * would need approximate nearest-neighbor for 10k+.
 */
function maximalMarginalRelevance(scored, user, { λ, targetCount }) {
  if (!scored.length) return [];

  const selected = [];
  const unselected = [...scored].sort((a, b) => b.finalScore - a.finalScore);
  const selectedEmbeddings = [];

  while (selected.length < targetCount && unselected.length > 0) {
    let bestIdx = 0;
    let bestMMR  = -Infinity;

    for (let i = 0; i < unselected.length; i++) {
      const candidate = unselected[i];
      const relevance = candidate.finalScore;

      // Maximum similarity to any already-selected item
      let maxSim = 0;
      if (selectedEmbeddings.length > 0 && candidate.post.embedding) {
        for (const selEmb of selectedEmbeddings) {
          const sim = cosineSimilarity(candidate.post.embedding, selEmb);
          if (sim > maxSim) maxSim = sim;
        }
      }

      const mmr = λ * relevance - (1 - λ) * maxSim * 1000;
      if (mmr > bestMMR) { bestMMR = mmr; bestIdx = i; }
    }

    const chosen = unselected.splice(bestIdx, 1)[0];
    selected.push(chosen);
    if (chosen.post.embedding) selectedEmbeddings.push(chosen.post.embedding);
  }

  return selected;
}

// ─── Boosted Content Injection ────────────────────────────────────────────────

function injectBoostedContent(feed, boosted) {
  if (!boosted.length) return feed;

  // Insert at deterministic positions (every N organic posts)
  const result = [];
  let boostIdx = 0;
  const boostInterval = Math.max(3, Math.floor(feed.length / (boosted.length + 1)));

  for (let i = 0; i < feed.length; i++) {
    if (boostIdx < boosted.length && i > 0 && i % boostInterval === 0) {
      result.push({ post: boosted[boostIdx++], finalScore: Infinity, boosted: true });
    }
    result.push(feed[i]);
  }
  return result;
}

// ─── Format Multipliers ───────────────────────────────────────────────────────

const FORMAT_MULTIPLIERS = {
  video_long:  1.4,   // most engaging by dwell time
  video_short: 1.2,   // high completion rate
  carousel:    1.1,
  image:       1.0,
  text_rich:   0.9,
  text_plain:  0.7,
  link:        0.8,   // leaves the app — mixed signal
};

// ─── Multi-Armed Bandit for Ad Targeting ─────────────────────────────────────

/**
 * Thompson Sampling for ad slot allocation.
 * Each ad maintains Beta distribution parameters (α, β) tracking success/failure.
 * Better than ε-greedy: explores intelligently rather than uniformly.
 */
export class ThompsonSamplingBandit {
  constructor(numArms) {
    this.alpha = new Float64Array(numArms).fill(1);  // successes + 1
    this.beta  = new Float64Array(numArms).fill(1);  // failures + 1
    this.n = numArms;
  }

  selectArm() {
    // Sample from Beta(α, β) for each arm, pick max
    const samples = Array.from({ length: this.n }, (_, i) =>
      this._betaSample(this.alpha[i], this.beta[i])
    );
    return samples.indexOf(Math.max(...samples));
  }

  update(arm, reward) {
    // reward ∈ {0, 1} for click/no-click, or continuous for revenue
    if (reward > 0) this.alpha[arm] += reward;
    else this.beta[arm] += 1 - reward;
  }

  getEstimates() {
    return Array.from({ length: this.n }, (_, i) => ({
      arm: i,
      mean: this.alpha[i] / (this.alpha[i] + this.beta[i]),
      uncertainty: Math.sqrt(
        this.alpha[i] * this.beta[i] /
        ((this.alpha[i] + this.beta[i])**2 * (this.alpha[i] + this.beta[i] + 1))
      ),
    }));
  }

  // Approximation via inverse transform (Johnk's method)
  _betaSample(α, β) {
    // For α=1,β=1 (uniform): trivially Math.random()
    // General: use rejection sampling or approximation
    if (α === 1 && β === 1) return Math.random();
    // Cheng's method for general α, β > 1
    if (α > 1 && β > 1) {
      const μ = α / (α + β);
      const λ = Math.sqrt(α * β / ((α + β + 1) * (α + β)**2));
      while (true) {
        const x = μ + λ * Math.tan(Math.PI * (Math.random() - 0.5));
        if (x < 0 || x > 1) continue;
        const logAccept = (α-1)*Math.log(x) + (β-1)*Math.log(1-x)
          - (α-1)*Math.log(μ) - (β-1)*Math.log(1-μ)
          - Math.log(1 + ((x-μ)/λ)**2);
        if (Math.log(Math.random()) < logAccept) return x;
      }
    }
    // Fallback: use normal approximation for large α, β
    const mean = α / (α + β);
    const variance = α * β / ((α + β)**2 * (α + β + 1));
    return Math.max(0, Math.min(1, mean + Math.sqrt(variance) * gaussianRandom()));
  }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom < 1e-10 ? 0 : dot / denom;
}

function gaussianRandom() {
  return Math.sqrt(-2 * Math.log(Math.random()+1e-15)) * Math.cos(2 * Math.PI * Math.random());
}
