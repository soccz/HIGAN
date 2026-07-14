export const meta = {
  name: 'tmlr-positioning-hunt',
  description: 'Ground HIGAN 3-pillar idea (exact JVP instrument / FD 2nd-order non-recoverability / curvature anchor-fragility) against top-tier prior art, then hunt TMLR positioning whitespace over 20 adversarial candidate rounds',
  phases: [
    { title: 'Literature sweep' },
    { title: 'Ground claims' },
    { title: 'Whitespace hunt' },
    { title: 'Synthesize' },
  ],
}

const IDEA = `PROJECT "HIGAN", targeting TMLR. Three-pillar contribution:
(P1) An EXACT composed forward-mode (JVP) INSTRUMENT for SECOND-ORDER geometry of deep generative models (StyleGAN1/2, Stable Diffusion). Computes per-direction second derivatives / extrinsic curvature ratio rho = ||d2G[v,v]|| / ||dG[v]|| in ~2x a generator forward pass, deterministic to ~17 digits. No retraining, no separate attribute classifier.
(P2) FINITE DIFFERENCES CANNOT RECOVER second-order generator geometry. Proposition: the naive central 2nd-difference (g(a+e)-2g(a)+g(a-e))/e^2 has an irreducible relative-error FLOOR ~ sqrt(eps_machine) from truncation-vs-cancellation tension. Empirics: on StyleGAN-bedroom the BEST fixed step still gives ~45 percent rel-error vs the exact JVP; fp64 does NOT remove the floor; a toy analytic referee confirms exact-JVP 2.5e-17 vs FD 8.6e-9. The exact composed-JVP instrument is the fix.
(P3) Using the instrument, curvature is ANCHOR-FRAGILE as a downstream signal. Single anchor: partial corr(curvature, edit-nonlinearity given first-order edit magnitude) = 0.45 (p=0.016, survives dropping outliers). Multi-anchor (5 seeds): partials [0.10, 0.19, -0.15, 0.37, 0.33], EVERY bootstrap 95 percent CI includes 0, sign even flips. So curvature weakly relates to edit nonlinearity but is NOT a robust, anchor-independent control signal. Crucially, the exact instrument is WHAT MAKES this fragility diagnosable at all (FD 45 percent floor would bury it).
DEAD lines that MUST NOT be revived (already killed by our own evidence): "rho predicts edit-risk in practice / is an actionable controller"; "FD robustly flips a downstream decision (sign-inversion)"; "we overturn Park et al." (Park NeurIPS23 is FIRST-order Jacobian-SVD only, computes no 2nd-order, nothing to overturn).
TMLR acceptance criteria (official): claims are accurate and supported by convincing evidence + some readers would learn something. SOTA/novelty NOT required.`

const LIT_SCHEMA = {
  type: 'object',
  properties: {
    cluster: { type: 'string' },
    papers: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' },
      venue: { type: 'string' },
      year: { type: 'string' },
      key_claim: { type: 'string' },
      derivative_method: { type: 'string', description: 'FD / autodiff / analytic / none; and what order of derivative' },
      relation_to_ours: { type: 'string', description: 'supports / threatens / orthogonal, and to which of P1/P2/P3' },
      top_tier: { type: 'boolean' },
    }, required: ['title', 'venue', 'key_claim', 'relation_to_ours'] } },
    biggest_threat: { type: 'string', description: 'the single strongest someone-already-did-this risk in this cluster' },
    open_gap: { type: 'string', description: 'clearest thing the literature has NOT done that we could' },
  },
  required: ['cluster', 'papers', 'biggest_threat', 'open_gap'],
}

const GROUND_SCHEMA = {
  type: 'object',
  properties: {
    pillar: { type: 'string' },
    closest_prior_art: { type: 'array', items: { type: 'object', properties: {
      paper: { type: 'string' }, how_close: { type: 'string' },
    }, required: ['paper', 'how_close'] } },
    is_novel: { type: 'boolean' },
    novelty_risk: { type: 'string' },
    overclaim_or_underclaim: { type: 'string' },
    airtight_scope: { type: 'string', description: 'the exact claim wording that stays defensible for TMLR' },
  },
  required: ['pillar', 'closest_prior_art', 'is_novel', 'airtight_scope'],
}

const CAND_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    one_line: { type: 'string' },
    framing: { type: 'string', description: 'how the paper is positioned under this lens' },
    pillars_used: { type: 'string' },
    why_whitespace: { type: 'string', description: 'what in the literature is unclaimed that this fills' },
    evidence_we_have: { type: 'string', description: 'which EXISTING evidence supports it; assume NO new experiments unless trivial' },
    section_role: { type: 'string', description: 'where in the paper this lives' },
  },
  required: ['title', 'one_line', 'framing', 'why_whitespace', 'evidence_we_have'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    tmlr_survival: { type: 'string', enum: ['strong', 'weak', 'dead'] },
    desk_reject_risk: { type: 'string' },
    strongest_objection: { type: 'string' },
    revives_dead_line: { type: 'boolean', description: 'true if it secretly resurrects rho-predicts / FD-decision-flip / Park-overturn' },
    fix: { type: 'string' },
    verdict_keep: { type: 'boolean' },
  },
  required: ['tmlr_survival', 'strongest_objection', 'verdict_keep'],
}

const SYNTH_SCHEMA = {
  type: 'object',
  properties: {
    literature_map: { type: 'string', description: 'where our work sits among the prior art; who are neighbors/competitors' },
    grounded_center: { type: 'string', description: 'the honest, defensible center of what we can claim' },
    top_whitespace: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, path: { type: 'string' }, why_tmlr: { type: 'string' }, risk: { type: 'string' },
    }, required: ['title', 'path', 'why_tmlr'] } },
    recommended_path: { type: 'string' },
    section_plan: { type: 'string' },
    key_risks: { type: 'string' },
  },
  required: ['literature_map', 'grounded_center', 'top_whitespace', 'recommended_path'],
}

const range = (n) => Array.from({ length: n }, (_, i) => i)
const gather = async (thunks) => (await parallel(thunks)).filter(Boolean)

// PHASE 1: LITERATURE SWEEP
phase('Literature sweep')
const CLUSTERS = [
  { key: 'A-higherorder-AD', desc: 'Higher-order and forward-mode automatic differentiation: JVP/HVP, Pearlmutter trick, nested / Taylor-mode AD, exact higher derivatives in deep nets and at scale. Venues: NeurIPS, ICML, JMLR, SISC, ACM TOMS, differentiable-programming workshops.' },
  { key: 'B-FD-numerics', desc: 'Numerical error of finite differences for SECOND derivatives: subtractive cancellation, optimal/adaptive step selection, complex-step differentiation, numerical differentiation of black-box / noisy functions. Venues: SIAM (SISC/SIMODS), Numerische Mathematik, ACM TOMS, J. Comput. Phys.' },
  { key: 'C-latent-geometry', desc: 'Riemannian / differential geometry of generative model latent spaces: pullback metric, curvature, geodesics, second fundamental form. Authors: Arvanitidis, Shao and Fletcher, Chen, Kuhnel, Yang, Park, Wang.' },
  { key: 'D-edit-directions', desc: 'Latent direction discovery / disentanglement / semantic editing where curvature or 2nd-order structure is invoked: GANSpace, SeFa, LatentCLR, DisCo, Park NeurIPS23, EditGAN, StyleSpace.' },
  { key: 'E-measurement-validity', desc: 'Measurement-validity papers, be-careful-with-this-metric / this-signal-is-confounded critiques, and PUBLISHABLE negative-result methodology in ML. What made them accepted, especially at TMLR / journals. e.g. critiques of disentanglement metrics, of FID, of probing, registered-report style nulls.' },
  { key: 'F-diffusion-geometry', desc: 'Diffusion-model latent / h-space geometry and differentiating the score / denoiser: score-Jacobian, Saito and Matsubara FD score-JVP, Asyrp/h-space, semantic latent directions in diffusion.' },
]
const lit = await gather(CLUSTERS.map((c) => () =>
  agent(
    'You are a literature scout for a TMLR submission. OUR IDEA:\n' + IDEA + '\n\nYOUR CLUSTER: ' + c.desc + '\n\nFind the strongest, most relevant TOP-TIER prior art in THIS cluster only. Try to load web tools first (ToolSearch query "select:WebSearch,WebFetch") and search for recent/specific papers; ALSO use your own knowledge (training cutoff 2026-01) which is rich for this area. For each paper give venue+year, its exact claim, whether it computes derivatives by FD vs autodiff vs analytic and AT WHAT ORDER, and precisely how it relates to / threatens / supports our pillars P1 (exact JVP instrument), P2 (FD cannot recover 2nd-order), P3 (curvature anchor-fragile). Identify the single biggest someone-already-did-this THREAT and the clearest OPEN GAP. Return 4-8 papers. Prioritize correctness over volume; do not invent papers; if unsure of a detail say so in the field.',
    { label: 'lit:' + c.key, phase: 'Literature sweep', schema: LIT_SCHEMA }
  )
))
const litStr = JSON.stringify(lit)
log('Literature: ' + lit.reduce((a, c) => a + (c.papers ? c.papers.length : 0), 0) + ' papers across ' + lit.length + ' clusters')

// PHASE 2: GROUND CLAIMS
phase('Ground claims')
const PILLARS = [
  { key: 'P1-instrument', text: 'P1 - the EXACT composed forward-mode JVP instrument for 2nd-order generator geometry (~2x forward, deterministic, no retraining).' },
  { key: 'P2-FD-floor', text: 'P2 - finite differences CANNOT recover 2nd-order generator geometry (sqrt(eps_mach) floor; 45 percent best-step error on StyleGAN; fp64 does not fix; toy analytic referee).' },
  { key: 'P3-anchor-fragile', text: 'P3 - curvature is ANCHOR-FRAGILE as a downstream signal (single-anchor partial 0.45 -> multi-anchor CIs all include 0, sign flips); instrument is what makes the fragility diagnosable.' },
]
const grounded = await gather(PILLARS.map((p) => () =>
  agent(
    'You are an adversarial novelty referee for a TMLR submission. OUR IDEA:\n' + IDEA + '\n\nLITERATURE MAP (JSON):\n' + litStr + '\n\nGROUND THIS PILLAR against the literature: ' + p.text + '\n\nWho is the CLOSEST prior art (be specific)? Are we genuinely novel or is this essentially known? Where do we OVER-claim or UNDER-claim relative to what is published? Give the exact AIRTIGHT scope wording that stays defensible. Be ruthless about the already-done risk.',
    { label: 'ground:' + p.key, phase: 'Ground claims', schema: GROUND_SCHEMA }
  )
))
const groundStr = JSON.stringify(grounded)

// PHASE 3: WHITESPACE HUNT (20 candidates, 5 lenses x 4 rounds)
phase('Whitespace hunt')
const LENSES = [
  'INSTRUMENT-FIRST: a measurement-tool paper where the exact JVP instrument is the primary contribution and the geometry findings are demonstrations of the tool.',
  'NUMERICAL-ANALYSIS CROSSOVER: the FD non-recoverability (sqrt(eps_mach) floor) as the headline result, with deep generative models as the high-impact application that makes it matter to ML.',
  'MEASUREMENT-VALIDITY WARNING: anchor-fragility as a cautionary methodology result - curvature-as-edit-signal is anchor-dependent; here is the protocol (multi-anchor + exact instrument) that exposes it.',
  'HONEST-NEGATIVE METHODOLOGY: what a publishable, decision-relevant NULL about generative-geometry control signals looks like, closing on a changed practice rather than self-negation.',
  'GENERALITY / CROSS-ARCHITECTURE: the instrument + the FD-floor + the fragility all transfer across StyleGAN and Stable Diffusion; that architecture-agnostic transfer is itself the contribution.',
]
const pool = []
for (const r of range(4)) {
  const seen = pool.map((c) => c.title).filter(Boolean).join(' | ') || '(none yet)'
  const fresh = await gather(LENSES.map((lens, i) => () =>
    agent(
      'You are a TMLR strategist hunting hidden whitespace for developing our idea. OUR IDEA:\n' + IDEA + '\n\nLITERATURE:\n' + litStr + '\n\nGROUNDED POSITION:\n' + groundStr + '\n\nALREADY-PROPOSED ANGLES (propose something genuinely DIFFERENT from these): ' + seen + '\n\nTHROUGH THIS LENS: ' + lens + '\n\nPropose ONE specific, concrete TMLR positioning / hidden-whitespace path that (a) the literature has NOT claimed, (b) our EXISTING evidence can support with no or only trivial new experiments, (c) TMLR would value (accurate claims backed by evidence + a reader takeaway). Be concrete about the headline sentence and which section each piece lives in. This is round ' + (r + 1) + ' of 4; push into less-obvious territory.',
      { label: 'gen:r' + (r + 1) + ':L' + (i + 1), phase: 'Whitespace hunt', schema: CAND_SCHEMA }
    )
  ))
  const judged = await gather(fresh.map((c) => () =>
    agent(
      'You are a skeptical TMLR action editor + cold reader. Adversarially evaluate this positioning candidate. OUR FIXED EVIDENCE is as described (assume NO new experiments beyond trivial reruns):\n' + IDEA + '\n\nCANDIDATE:\n' + JSON.stringify(c) + '\n\nWill it survive desk-screen AND review? What is the single strongest objection? Does it SECRETLY revive a dead line (rho-predicts-edit-risk / FD-decision-flip / Park-overturn)? Does it self-negate (we-find-no-use) in a way that loses the reader? Give a fix. Be calibrated: most candidates are weak. Verdict: strong/weak/dead and a keep boolean (keep only if strong, or weak-but-fixable-and-distinct).',
      { label: 'judge:r' + (r + 1) + ':' + (c.title || '').slice(0, 18), phase: 'Whitespace hunt', schema: VERDICT_SCHEMA }
    ).then((v) => ({ ...c, verdict: v }))
  ))
  pool.push(...judged)
  log('Round ' + (r + 1) + '/4: ' + judged.length + ' candidates judged; pool now ' + pool.length)
}
const kept = pool.filter((c) => c.verdict && c.verdict.verdict_keep)
const strong = pool.filter((c) => c.verdict && c.verdict.tmlr_survival === 'strong')
log('Whitespace hunt done: ' + pool.length + ' total, ' + kept.length + ' kept, ' + strong.length + ' strong')

// PHASE 4: SYNTHESIZE
phase('Synthesize')
const synth = await agent(
  'You are the lead researcher synthesizing a TMLR development strategy. OUR IDEA:\n' + IDEA + '\n\nLITERATURE MAP:\n' + litStr + '\n\nGROUNDED PILLARS:\n' + groundStr + '\n\nSURVIVING WHITESPACE CANDIDATES (with adversarial verdicts):\n' + JSON.stringify(kept) + '\n\nSTRONG candidates specifically:\n' + JSON.stringify(strong) + '\n\nProduce: (1) a literature_map placing our work among neighbors/competitors and naming the real threats; (2) the honest grounded_center of what we can claim; (3) top_whitespace = the 3-5 best hidden-space TMLR paths, ranked, each with the development path, why TMLR accepts it, and its risk; (4) a single recommended_path; (5) a section_plan for that path; (6) key_risks. Be concrete and decision-ready; this feeds the lead final call.',
  { phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return {
  n_papers: lit.reduce((a, c) => a + (c.papers ? c.papers.length : 0), 0),
  grounded,
  n_candidates: pool.length,
  n_kept: kept.length,
  n_strong: strong.length,
  kept,
  synthesis: synth,
}
