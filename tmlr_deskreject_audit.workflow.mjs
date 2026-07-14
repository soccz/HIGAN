export const meta = {
  name: 'tmlr-deskreject-audit',
  description: '20-lens adversarial audit of tmlr_predict.tex against the TMLR desk-reject guides, then adversarially re-check every flagged risk and synthesize a desk-pass verdict',
  phases: [
    { title: '20-lens audit' },
    { title: 'Adversarial re-check' },
    { title: 'Synthesize' },
  ],
}

const ART = '/home/soccz/22tb/study/HIGAN/note/submission/tmlr_predict.tex'
const G1 = '/home/soccz/22tb/.claude-packs/tmlr/guide/DESK_REJECT_AUTOPSY.md'
const G2 = '/home/soccz/22tb/.claude-packs/tmlr/guide/NEGATIVE_AUDIT_PLAYBOOK.md'
const G3 = '/home/soccz/22tb/.claude-packs/tmlr/guide/TMLR_WRITING_GUIDE.md'

const CTX = 'You are auditing a TMLR submission for DESK-REJECT risk. Manuscript: ' + ART + '. The binding standards are the project desk-reject guides; READ ALL THREE before judging: ' + G1 + ' , ' + G2 + ' , ' + G3 + ' . CRITICAL CONTEXT: this manuscript is the direct successor of paper "9732" dissected in DESK_REJECT_AUTOPSY.md (same author, was DESK-REJECTED). It was since heavily revised: title changed to a plain declarative, self-negation removed, FFHQ robustness added, numbered Recommendations 1-3 added, the case study reframed as a replication audit. Judge whether the CURRENT text still trips the guide desk-reject triggers. Quote the guide rule and the manuscript line as evidence. Be a harsh EIC, not a cheerleader, but do not invent problems the text does not have.'

const AUDIT_SCHEMA = { type:'object', properties:{
  lens:{type:'string'},
  verdict:{type:'string', enum:['PASS','RISK','BLOCKER']},
  finding:{type:'string'},
  guide_basis:{type:'string', description:'which guide rule (quoted) this maps to'},
  manuscript_evidence:{type:'string', description:'the line or phrase in the manuscript'},
  desk_reject_risk:{type:'string', enum:['low','medium','high']},
  fixable_without_new_experiments:{type:'boolean'},
  fix:{type:'string'},
}, required:['lens','verdict','finding','desk_reject_risk','fix'] }

const RECHECK_SCHEMA = { type:'object', properties:{
  lens:{type:'string'},
  original_verdict:{type:'string'},
  holds_up:{type:'boolean', description:'does the flagged risk survive skeptical scrutiny'},
  is_real_desk_reject_trigger:{type:'boolean'},
  severity:{type:'string', enum:['cosmetic','minor','serious','fatal']},
  reasoning:{type:'string'},
  recommended_action:{type:'string'},
}, required:['lens','holds_up','is_real_desk_reject_trigger','severity','recommended_action'] }

const SYNTH_SCHEMA = { type:'object', properties:{
  desk_pass_estimate:{type:'string', description:'likelihood the manuscript clears EIC desk screen, with reasoning grounded in the guides'},
  blockers:{type:'array', items:{type:'object', properties:{ issue:{type:'string'}, why:{type:'string'}, fix:{type:'string'} }, required:['issue','fix']}},
  strong_points:{type:'array', items:{type:'string'}},
  prioritized_fixes:{type:'array', items:{type:'string'}},
  verdict:{type:'string', enum:['SUBMIT','FIX-THEN-SUBMIT','MAJOR-REWRITE','WRONG-VENUE']},
}, required:['desk_pass_estimate','blockers','prioritized_fixes','verdict'] }

const gather = async (thunks) => (await parallel(thunks)).filter(Boolean)

phase('20-lens audit')
const LENSES = [
  {k:'pattern8-consequence', q:'Pattern 8 (the trigger that killed 9732): is every negative closed by a CHANGED DECISION? Are the numbered Recommendations 1-3 (use exact / report step axis / replicate across anchors) as strong as the accepted-7 consequence moves (olGaiwoZHZ impose-structure-accuracy-improves, Pitfalls researchers-should), or is there still NO positive-lever DEMONSTRATION, only a prescription? Is the missing FD-to-exact-changes-a-downstream-result demonstration fatal, or do the numbered prescriptions satisfy the guide?'},
  {k:'self-negation', q:'Pattern 1 self-negating contribution: any residue of field-is-mostly-safe / most-methods-avoid-the-regime / no-downstream-win?'},
  {k:'self-deprecation', q:'Pattern 2 self-deprecation: any note/protocol/textbook/audit/measurement-study framing that demotes the contribution below the bar? Is a deterministic measurement study or measurement referee a problem given the guide?'},
  {k:'living-positive', q:'Move 6: is the strong negative separated from a LIVING POSITIVE? Does the replication audit (anchor-fragility) read as curvature-is-not-useful-use-first-order (self-negation like 9736 APAT-does-not-beat-never-use), or is the instrument value made the positive?'},
  {k:'title-30s', q:'Title: does Finite Differences Are Unreliable for Second-Order Generator Geometry pass the guide title formula (plain declarative finding, no gerund, under 20 words, no jargon front-load)? Compare to guide exemplars and the dead 9732 title.'},
  {k:'abstract-30s', q:'Abstract 30s test: do the first two sentences pin who and what (not jargon or numbers)? Is the closing a confident finding plus living positive or operational recommendation? Any desk-instant-death phrase?'},
  {k:'llm-cadence', q:'Pattern 3 LLM-cadence: dense X-not-Y parallelism, per-sentence parenthetical hedges, coined-then-bounded terms, self meta-commentary. Quote any you find.'},
  {k:'private-ontology', q:'Pattern 4 private ontology: coined abbreviations or terms; is trilemma defined and is the coined-term count within umbrella-1 or at most 3?'},
  {k:'audience-present', q:'Move 3 / Pattern 6: is the at-risk reader a PRESENT-TENSE concrete practitioner (geodesic, curvilinear, Hessian-Penalty users), with zero in-the-future-someone framing?'},
  {k:'real-system', q:'Move 4: are real in-use systems audited (StyleGAN bedroom and FFHQ, Hessian Penalty) with zero self-built strawman? Is the NL metric or HiGAN-boundary setup a strawman risk?'},
  {k:'competent-ref', q:'Move 4 and 5: does a competent reference SUCCEED where FD fails (the exact JVP and the first-order control), proving the failure belongs to the estimator and not the task?'},
  {k:'apparatus-size', q:'Pattern 7 / Move 5: is the apparatus smaller than the payload and under-claimed, or is heavy machinery over-packaged on thin evidence?'},
  {k:'effect-size-assert', q:'Move 6: is the strong negative (trilemma) asserted with effect sizes rather than buried in hedges, while only the genuinely weak cell (the anchor case study) is qualified? Is hedging calibrated to evidence strength?'},
  {k:'limitations-cohesion', q:'Move 7: are limitations cohered in one section and closed with a-fortiori or scope statements, not sprinkled as disclaimers across abstract, intro, and conclusion?'},
  {k:'claim-evidence', q:'Criterion 1: is every claim matched to evidence with no bare claims; do body numbers (trilemma 3.0/2.0/0.2, FFHQ 5.0/1.5/0.x, pooled p 0.002, CI spans 0) match the tables; is any claim over-scoped beyond its evidence?'},
  {k:'9732-recurrence', q:'Direct 1:1 against the 8 patterns that killed 9732 in DESK_REJECT_AUTOPSY: which are now fixed, which (if any) recur even partially? Be specific per pattern number.'},
  {k:'eic-desk-30s', q:'Simulate the EIC 30-second desk read (title, then abstract first two sentences, then contributions skim). Does it read as a CONTRIBUTION or a note? Would it be desk-rejected? One-line EIC verdict.'},
  {k:'consequence-real-vs-claimed', q:'Is the consequence DEMONSTRATED or merely ASSERTED? The Hessian-Penalty inherits-the-floor claim: is it shown to change that method result, or only argued? Given the guide allows closing consequence by numbered prescriptions (Pitfalls), is the numbered-recommendation route sufficient here without a positive-lever demonstration?'},
  {k:'note-vs-paper', q:'Guide reads-as-a-note risk: the core math is textbook numerical analysis (sqrt-eps floor, Higham/Press) and the tool is standard AD. Does the trilemma plus FFHQ plus replication audit lift it above tutorial/note, or does it still read as a known-result note?'},
  {k:'cold-read-whole', q:'As a fresh hostile TMLR reviewer with no author context: overall lean (accept / weak-accept / borderline / reject), the single strongest objection, and whether you would push to desk-reject or send to review.'},
]
const audit = await gather(LENSES.map((l, i) => () =>
  agent(CTX + '\n\nYOUR LENS (#' + (i + 1) + '): ' + l.q, { label: 'audit:' + l.k, phase: '20-lens audit', schema: AUDIT_SCHEMA })
))
const flagged = audit.filter(a => a.verdict && a.verdict !== 'PASS')
log('Audit: ' + audit.length + ' lenses, ' + flagged.length + ' flagged (RISK/BLOCKER)')

phase('Adversarial re-check')
const rechecked = await gather(flagged.map(f => () =>
  agent('You are a SKEPTICAL second reviewer re-checking a flagged desk-reject risk on the TMLR manuscript ' + ART + ' (guides: ' + G1 + ', ' + G2 + ', ' + G3 + '). A first auditor flagged this:\n' + JSON.stringify(f) + '\n\nRe-read the relevant manuscript passage and the cited guide rule. Does this risk SURVIVE scrutiny, or did the auditor over-call it? Is it a REAL desk-reject trigger or cosmetic? Rate severity and give the action. Default to skepticism: most flags are not actually desk-fatal.',
    { label: 'recheck:' + (f.lens || '').slice(0, 18), phase: 'Adversarial re-check', schema: RECHECK_SCHEMA })
))

phase('Synthesize')
const synth = await agent('You are the lead deciding whether to submit to TMLR or fix first. Manuscript: ' + ART + '. Full 20-lens audit:\n' + JSON.stringify(audit) + '\n\nAdversarial re-checks of flagged items:\n' + JSON.stringify(rechecked) + '\n\nProduce: desk_pass_estimate (reasoning grounded in the guides), the real blockers (only those that survived re-check as serious or fatal), strong_points, a prioritized fix list, and a single verdict (SUBMIT / FIX-THEN-SUBMIT / MAJOR-REWRITE / WRONG-VENUE). Be decisive and honest; the author keeps getting desk-rejected and needs the truth, not reassurance.',
  { phase: 'Synthesize', schema: SYNTH_SCHEMA })

return { lenses: audit.length, flagged: flagged.length, audit, rechecked, synthesis: synth }
