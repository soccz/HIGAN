export const meta = {
  name: 'consequence-hunt',
  description: 'Hunt for a real downstream consequence where a finite-difference step choice flips a magnitude/sign-dependent decision that the exact instrument gets right, then adversarially verify each candidate against the existing evidence and the project deflationary prior',
  phases: [
    { title: 'Candidate generation' },
    { title: 'Adversarial data/feasibility check' },
    { title: 'Synthesize' },
  ],
}

const CTX = 'CONTEXT. HIGAN TMLR paper: an exact composed-JVP reference vs central finite differences for second-order generator curvature on StyleGAN/FFHQ. The trilemma is established (FD steps for magnitude/sign/rank optimize at different points). BUT every downstream test so far shows FD TRACKS exact, so the paper has NO demonstrated downstream consequence: ranking Spearman(exact,FD)=0.82-0.96, top-3 selection FD@delta=1.0 = exact (0.667), prediction rho_FD 0.19-0.35 vs rho_exact 0.47, and the committed evidence files store verdict_consequence_demonstrated=false / verdict_direction_flip=false. The ONE axis where FD genuinely fails is MAGNITUDE (>=45% floor, fp64 does not fix) and SIGN (bias -26% to +350%). Rank is recovered by FD, so a consequence CANNOT live on the ranking axis. Evidence dir: /home/soccz/22tb/study/HIGAN/note/submission/evidence/ (stylegan_fd_vs_exact_table2.json has per-direction c_exact/c_fd2 over 16 steps; curvature_decision_gt_metrics.json, curvature_ordering_flip_metrics.json). Scripts: /home/soccz/22tb/study/HIGAN/higan_dev/. PROJECT PRIOR (critical): this line is deflationary -- strong signals have repeatedly turned out to be artifacts; assume any proposed consequence is FAKE until the data forces otherwise.'

const CAND_SCHEMA = { type:'object', properties:{
  decision:{type:'string', description:'the concrete downstream decision a practitioner makes from second-order curvature'},
  why_magnitude_or_sign:{type:'string', description:'why this decision depends on curvature MAGNITUDE or SIGN (not ranking, where FD is fine)'},
  how_fd_step_could_flip:{type:'string', description:'the specific mechanism by which a finite-difference step choice would give a different answer than exact'},
  literature_anchor:{type:'string', description:'a real published method/use that makes this decision (e.g. Hessian Penalty, geodesic energy)'},
  testable_from_existing_data:{type:'boolean'},
  experiment_if_new:{type:'string', description:'if not testable from existing artifacts, the minimal experiment that would show the flip'},
  honest_flip_likelihood:{type:'string', enum:['high','medium','low','probably-fake'], description:'given FD tracks exact everywhere so far, how likely is a REAL flip'},
}, required:['decision','why_magnitude_or_sign','how_fd_step_could_flip','testable_from_existing_data','honest_flip_likelihood'] }

const CHECK_SCHEMA = { type:'object', properties:{
  decision:{type:'string'},
  checked_how:{type:'string', description:'what data/computation was actually run or inspected'},
  flip_found:{type:'boolean', description:'does a fixed-step FD actually flip this decision vs exact'},
  numbers:{type:'string', description:'the concrete numbers, with file:provenance'},
  verdict:{type:'string', enum:['real-consequence','no-flip-fd-tracks-exact','needs-new-gpu-experiment','artifact-risk']},
  honest_note:{type:'string'},
}, required:['decision','flip_found','verdict','honest_note'] }

const SYNTH_SCHEMA = { type:'object', properties:{
  any_real_consequence:{type:'boolean'},
  best_candidate:{type:'string'},
  what_the_data_supports:{type:'string'},
  recommended_experiment:{type:'string', description:'the single experiment most likely to yield a real consequence, or NONE if structurally absent'},
  honest_verdict:{type:'string', description:'is there a findable consequence, or is the line structurally consequence-free -> numerics venue'},
}, required:['any_real_consequence','what_the_data_supports','honest_verdict'] }

const gather = async (thunks) => (await parallel(thunks)).filter(Boolean)

phase('Candidate generation')
const LENSES = [
  'Hessian-Penalty-style training regularizers: where curvature MAGNITUDE sets a loss/regularizer weight, so a wrong magnitude changes what gets penalized and thus the trained model or a stop/continue decision.',
  'Geodesic / path-length decisions: where the integrated curvature magnitude decides which of two edit paths is shorter/straighter, so a 45% magnitude error could flip the choice between two close paths.',
  'Thresholded go/no-go on curvature magnitude or sign: e.g. "is this edit direction safe (low curvature)?" or "is the second-order term negligible here?" where a fixed threshold flips under FD step.',
  'Sign-dependent decisions: where the SIGN of a second-order quantity (concavity, attraction vs repulsion of a flow, sign of a curvature-correction term) decides an action, and the FD signed bias (-26% to +350%, crossing zero) flips it.',
  'Model/architecture comparison: ranking two GENERATORS or two layers by curvature magnitude for a design decision, where FD magnitude error reorders the comparison even though within-generator direction ranking is preserved.',
  'Adversarial / sanity lens: argue the OPPOSITE -- that for any magnitude-dependent decision, FD error is a smooth monotone scaling that preserves the decision boundary, so no flip exists; identify what would have to be true (non-monotone FD error across the decision set) for a flip, and whether the data shows it.',
]
const candidates = (await gather(LENSES.map((l,i) => () =>
  agent(CTX + '\n\nLENS ' + (i+1) + ': ' + l + '\n\nPropose 1-2 concrete downstream decisions under this lens where a finite-difference STEP choice would flip the answer vs the exact instrument, on the MAGNITUDE or SIGN axis only (ranking is off-limits -- FD is fine there). For each, be specific about the mechanism and whether it is checkable from the existing evidence JSONs or needs a new experiment. Rate honest_flip_likelihood pessimistically: the default truth in this project is FD tracks exact.',
    {label:'gen:L'+(i+1), phase:'Candidate generation', schema:CAND_SCHEMA})
))).flat?.() || []
const cand = candidates
log('Generated ' + cand.length + ' consequence candidates')

phase('Adversarial data/feasibility check')
const checked = await gather(cand.map(c => () =>
  agent(CTX + '\n\nA candidate downstream consequence was proposed:\n' + JSON.stringify(c) + '\n\nYour job: SKEPTICALLY verify whether a fixed-step finite difference ACTUALLY flips this decision vs exact. If testable from existing artifacts, run python/jq against the evidence JSONs in /home/soccz/22tb/study/HIGAN/note/submission/evidence/ (and higan_dev/out if needed) and report the real numbers with provenance. If it needs a new experiment, say exactly what and rate the risk it will just reproduce FD-tracks-exact. DEFAULT TO no-flip: only return flip_found=true if the numbers genuinely show a decision reversing. Do not manufacture a flip; the project has already been burned by a forced consequence that contradicted its own data.',
    {label:'check:'+(c.decision||'').slice(0,20), phase:'Adversarial data/feasibility check', schema:CHECK_SCHEMA})
))

phase('Synthesize')
const synth = await agent(CTX + '\n\nAll candidates:\n' + JSON.stringify(cand) + '\n\nAdversarial checks:\n' + JSON.stringify(checked) + '\n\nHonestly synthesize: is there ANY real downstream consequence (a fixed-step FD flips a magnitude/sign decision the exact tool gets right), backed by data or a credible cheap experiment? Name the best candidate and what the data actually supports. If every candidate collapses to FD-tracks-exact, say so plainly -- the line is structurally consequence-free and belongs at a numerics/AD venue, not forced into TMLR. Do not be optimistic to please; the author needs the truth.',
  {phase:'Synthesize', schema:SYNTH_SCHEMA})

return { n_candidates: cand.length, candidates: cand, checked, synthesis: synth }
