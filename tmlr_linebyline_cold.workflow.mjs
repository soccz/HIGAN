export const meta = {
  name: 'tmlr-linebyline-cold',
  description: 'Context-free hostile line-by-line audit of tmlr_predict.tex: each chunk read by a fresh anchor-free reviewer instructed to assume the paper is over-polished to hide weakness, then adversarial re-check and an honest cold verdict',
  phases: [
    { title: 'Line-by-line cold audit' },
    { title: 'Re-check serious findings' },
    { title: 'Honest cold verdict' },
  ],
}

const ART = '/home/soccz/22tb/study/HIGAN/note/submission/tmlr_predict.tex'

const CTX = 'You are a HOSTILE, skeptical TMLR reviewer (for some lenses, the Editor-in-Chief) seeing this manuscript for the FIRST time, with ZERO author context and ZERO knowledge of any prior review, gate, or revision. Manuscript: ' + ART + '. DEFAULT STANCE: this paper is probably over-polished to hide a weakness; your job is to find what is wrong, not to confirm it is fine. The author has a documented history of desk-rejection for over-hedged, AI-textured, self-deflating prose and for measurement papers with no real downstream consequence; assume those habits are present until you prove them absent line by line. Polish is the trap, not the evidence. Quote the EXACT line for every issue. Do not be charitable.'

const AUDIT_SCHEMA = { type:'object', properties:{
  chunk:{type:'string'},
  issues:{type:'array', items:{type:'object', properties:{
    quote:{type:'string', description:'the exact offending line or phrase'},
    type:{type:'string', enum:['claim-not-supported','overclaim','manufactured-hedge','ai-cadence','logical-leap','vague','contradiction','self-deflation','consequence-missing','other']},
    severity:{type:'string', enum:['nit','minor','serious','fatal']},
    why:{type:'string'},
  }, required:['quote','type','severity','why']}},
  ai_cadence_score:{type:'integer', description:'0=clearly human, 10=obviously AI-generated texture'},
  worst_issue:{type:'string'},
}, required:['chunk','issues','ai_cadence_score','worst_issue'] }

const RECHECK_SCHEMA = { type:'object', properties:{
  issue:{type:'string'},
  survives:{type:'boolean', description:'does it survive a skeptical second look, or was the reviewer nitpicking'},
  real_severity:{type:'string', enum:['nit','minor','serious','fatal']},
  is_desk_or_review_killer:{type:'boolean'},
  reasoning:{type:'string'},
  fix:{type:'string'},
}, required:['issue','survives','real_severity','is_desk_or_review_killer','fix'] }

const SYNTH_SCHEMA = { type:'object', properties:{
  honest_cold_verdict:{type:'string', description:'what a genuinely cold EIC/reviewer concludes, no anchoring'},
  prior_gates_were_anchor_fooled:{type:'boolean'},
  anchor_fooled_explanation:{type:'string'},
  real_blockers:{type:'array', items:{type:'object', properties:{ issue:{type:'string'}, severity:{type:'string'}, fix:{type:'string'} }, required:['issue','fix']}},
  ai_cadence_overall:{type:'string'},
  desk_pass_honest_estimate:{type:'string'},
  top_fixes:{type:'array', items:{type:'string'}},
  decision:{type:'string', enum:['SUBMIT','FIX-THEN-SUBMIT','MAJOR-REWRITE']},
}, required:['honest_cold_verdict','prior_gates_were_anchor_fooled','real_blockers','desk_pass_honest_estimate','decision'] }

const gather = async (thunks) => (await parallel(thunks)).filter(Boolean)

phase('Line-by-line cold audit')
const CHUNKS = [
  {k:'abstract', lines:'30-60'},
  {k:'intro-motivation-trilemma', lines:'62-87'},
  {k:'intro-fix-and-contributions', lines:'89-138'},
  {k:'setup', lines:'140-159'},
  {k:'prop1-and-toy', lines:'161-223'},
  {k:'sec-real-trilemma-and-table', lines:'225-298'},
  {k:'instrument', lines:'300-319'},
  {k:'replication-audit', lines:'321-381'},
  {k:'recommendations-scope-limits', lines:'383-427'},
  {k:'related-work', lines:'429-450'},
  {k:'conclusion', lines:'452-467'},
  {k:'reproducibility-and-appendix', lines:'469-589'},
]
const chunkAudits = await gather(CHUNKS.map(c => () =>
  agent(CTX + '\n\nRead lines ' + c.lines + ' of the manuscript (use Read with offset/limit). Audit EVERY sentence in this passage line by line. For each problem report the exact quote, type, severity, and why: (a) claim NOT supported by evidence actually shown in the paper; (b) overclaim beyond what the numbers support; (c) a manufactured hedge or scope-qualifier that needlessly weakens a claim the paper could state outright; (d) AI-cadence — X-not-Y parallelism, parenthetical-hedge pileup, coined-then-bounded terms, self-meta-commentary like "the lesson is" / "reported at the strength its evidence supports" / "is itself a result" — and answer in ai_cadence_score whether THIS passage reads as AI-written; (e) logical leap / non-sequitur; (f) vague or empty phrase; (g) a number or claim that contradicts another part of the paper. Be exhaustive and harsh; list nits too but mark them nit. If the passage is genuinely clean, say so but only after a real line-by-line pass.',
    {label:'chunk:'+c.k, phase:'Line-by-line cold audit', schema:AUDIT_SCHEMA})
))

const LENSES = [
  {k:'ai-cadence-whole', q:'Read the WHOLE manuscript. The single question: does this read as AI-generated / over-LLM-polished text? Hunt the tells across the entire paper — X-not-Y parallel constructions, em-dash interruptions, parenthetical hedges, coined-then-immediately-bounded phrasing, self-referential meta-commentary, suspiciously uniform sentence rhythm. Score it and quote the worst 5 offenders with line.'},
  {k:'eic-cold-30s', q:'You are the EIC. Read ONLY the title, the first two sentences of the abstract, and the contribution list — nothing else. In 30 seconds: is this a contribution or a note? Send to review or desk-reject? Quote what triggers your verdict. Do not read the body; the EIC does not.'},
  {k:'strongest-objection', q:'Read the whole paper. As the harshest possible reviewer, what is the SINGLE strongest objection that could sink this paper at review? State it as a reviewer would write it, with the specific lines/claims it targets. Then state honestly whether it is fatal, serious, or survivable.'},
  {k:'consequence-reality', q:'Read the whole paper. The paper claims a downstream consequence (the Hessian Penalty inherits the floor; the exact instrument changes the audit conclusion). Is this consequence actually DEMONSTRATED, or only asserted/argued? Does the paper ever show one concrete case where finite differences and the exact instrument yield DIFFERENT downstream conclusions? If not, quote the conditional language ("could", "cannot be trusted", "would") that stands in for a demonstration.'},
  {k:'claim-evidence-numbers', q:'Read the results sections and tables (sec-real, replication-audit, appendix). Check internal consistency of the numbers: do the abstract/intro figures match the tables? Does the trilemma narrative (mag-best 3.0, bias-best 2.0, rank-best 0.2) match Table 2 and the appendix? Does the audit (single +0.45 -> cross +0.17, CI spans 0, pooled p~0.002) cohere? Flag any number that is stated but not backed by a visible table/figure, or any mismatch.'},
  {k:'note-vs-paper', q:'Read the whole paper. The core math is a textbook sqrt(eps_mach) finite-difference floor (Higham/Press) and the tool is standard AD. Strip away the framing: is the actual NEW content enough to be a paper, or is it a known-result note dressed up? What, concretely, is novel here that a numerical-analysis reviewer would not call textbook? Be brutal.'},
]
const lensAudits = await gather(LENSES.map(l => () =>
  agent(CTX + '\n\nYOUR TASK: ' + l.q, {label:'lens:'+l.k, phase:'Line-by-line cold audit', schema:AUDIT_SCHEMA})
))

const all = chunkAudits.concat(lensAudits)
const serious = []
all.forEach(a => (a.issues||[]).forEach(i => { if (i.severity === 'serious' || i.severity === 'fatal') serious.push({from:a.chunk, ...i}) }))
log('Cold audit: ' + all.length + ' passes, ' + serious.length + ' serious/fatal issues raised')

phase('Re-check serious findings')
const rechecked = await gather(serious.map(s => () =>
  agent('You are a SKEPTICAL adjudicator. A hostile cold reviewer raised this issue on the TMLR manuscript ' + ART + ':\n' + JSON.stringify(s) + '\n\nRead the actual passage. Does the issue SURVIVE scrutiny, or was the reviewer over-hostile / nitpicking / wrong? Rate the REAL severity and whether it would actually kill the paper at desk or review. Give the concrete fix if it survives. Be fair: hostile reviewers over-call, but do not whitewash a real problem.',
    {label:'recheck:'+(s.type||'').slice(0,16), phase:'Re-check serious findings', schema:RECHECK_SCHEMA})
))

phase('Honest cold verdict')
const synth = await agent('You are delivering an honest COLD desk-pass verdict on this just-revised TMLR manuscript; the author needs to know if it now clears roughly 70 percent desk-pass, stated plainly and calibrated, not reassuring. Manuscript: ' + ART + '. Full cold line-by-line audit (12 chunks + 6 lenses):\n' + JSON.stringify(all) + '\n\nAdversarial re-checks of the serious/fatal issues:\n' + JSON.stringify(rechecked) + '\n\nDeliver: honest_cold_verdict (what a genuinely cold EIC/reviewer concludes); whether the prior all-PASS gates were likely anchor-fooled and why; the REAL blockers that survived re-check (serious/fatal only); the overall AI-cadence read; an honest desk-pass estimate (calibrated, not reassuring); the top fixes; and a decision. The author explicitly does not want reassurance — give the truth even if it contradicts the earlier "80-85% desk-pass / GO" conclusions.',
  {phase:'Honest cold verdict', schema:SYNTH_SCHEMA})

return { passes: all.length, serious: serious.length, chunkAudits, lensAudits, rechecked, synthesis: synth }
