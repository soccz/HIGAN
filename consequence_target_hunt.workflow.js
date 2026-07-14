export const meta = {
  name: 'fd9732-consequence-target-hunt',
  description: '9732 ⓐ경로(TMLR): published curvature/latent-geometry pipeline 중 (1)정성 결론을 내고 (2)그 결론이 2차 기하량(curvature/Christoffel/geodesic/2nd-pushforward)에 의존하며 (3)재현가능한 후보를 찾는다. exact-AD vs /eps^2-FD로 재계산하면 정성 결론이 뒤집히는(consequence) 단일 타겟 1개 추천, 또는 없으면 정직하게 ⓑ(수치해석 venue) 회귀 판정. 멀티모달 디스커버리(로컬 survey + web) → 후보별 적합성 평가 → 랭킹+실험설계.',
  phases: [
    { title: 'Discover', detail: '로컬 survey audit + web(diffusion/GAN/VAE latent-geometry) 멀티모달 후보 수집' },
    { title: 'Assess', detail: '후보별 2차-의존성·정성결론·재현성·flip가능성 평가(web 검증)' },
    { title: 'Rank', detail: '단일 타겟 추천 + 실험설계, 또는 ⓑ 회귀 판정 + 완전성 비평' },
  ],
}

const DIR = '/home/soccz/22tb/study/HIGAN'
const PLAYBOOK = '/home/soccz/22tb/.claude-packs/tmlr/guide/NEGATIVE_AUDIT_PLAYBOOK.md'

// 핵심 판정 기준 (모든 에이전트에 주입)
const CRITERIA = `[ⓐ 타겟 적합 3대 조건 — 모두 만족해야 함]
1. QUALITATIVE-CONCLUSION: 그 논문이 *정성적 결론*을 낸다 — attribute/sample/layer를 곡률로 ordering, clustering, "A가 B보다 더 curved", geodesic interpolation 품질 비교, curvature 기반 정성 판단 등. (단순 정량표만이면 약함)
2. SECOND-ORDER-DEPENDENT: 그 결론이 *2차 기하량*에 의존한다 — Riemann/sectional/scalar curvature, Christoffel symbol, geodesic(2차 ODE), Hessian/2nd pushforward. **1차 metric pullback(G=J^T J, magnification factor, Jacobian only)만 쓰면 제외** — 1차 FD는 같은 코드에서 2.4%로 안전(C11)이라 floor에 안 걸린다.
3. REPRODUCIBLE: 코드/가중치/데이터가 공개되어 *우리가 bounded하게* 재계산 가능(이상적: github + pretrained generator). 큰 학습 불필요.
[consequence의 의미] 같은 pipeline을 (i) central /eps^2 FD(실무자가 자연히 쓰는 route)와 (ii) exact composed-AD JVP로 곡률을 재계산했을 때, 정성 결론(ordering/which-is-more-curved/clustering)이 *뒤집히면* = TMLR을 살리는 "틀린 측정 → 바뀌는 결정". 원논문이 FD를 썼을 필요는 없다 — "naive FD route라면 결론 B, 진짜는 A"면 충분.`

const CAND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['candidates'],
  properties: {
    candidates: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['name', 'ref', 'venue_year', 'qualitative_conclusion', 'geometric_quantity', 'order', 'how_computed', 'code_available', 'source', 'why_candidate'],
      properties: {
        name: { type: 'string', description: '논문/파이프라인 식별' },
        ref: { type: 'string', description: '저자+제목 또는 url/arxiv id' },
        venue_year: { type: 'string' },
        qualitative_conclusion: { type: 'string', description: '그 논문이 내는 정성 결론(없으면 "none/quantitative-only")' },
        geometric_quantity: { type: 'string', description: '핵심 기하량(curvature/Christoffel/geodesic/metric/...)' },
        order: { type: 'string', description: 'first(1차 metric only) | second(2차 곡률/geodesic) | unclear' },
        how_computed: { type: 'string', description: 'analytic/AD/FD/unknown — 곡률을 어떻게 계산하는가' },
        code_available: { type: 'string', description: 'yes(+url) / no / unknown' },
        source: { type: 'string', description: 'local-survey | web-search | web-fetch' },
        why_candidate: { type: 'string', description: '왜 ⓐ 후보인지 1줄' },
      } } },
  },
}

const ASSESS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name', 'cond1_qualitative', 'cond2_second_order', 'cond3_reproducible', 'flip_plausibility', 'fit_for_a', 'experiment_sketch', 'notes'],
  properties: {
    name: { type: 'string' },
    cond1_qualitative: { type: 'string', description: 'PASS/FAIL + 어떤 정성 결론인지' },
    cond2_second_order: { type: 'string', description: 'PASS/FAIL + 어떤 2차 기하량에 의존하는지(1차뿐이면 FAIL)' },
    cond3_reproducible: { type: 'string', description: 'PASS/PARTIAL/FAIL + 코드/가중치/데이터 가용성 + 우리 작업량 추정' },
    flip_plausibility: { type: 'string', description: 'high/med/low + 근거: FD floor가 이 결론을 뒤집을 만큼 큰가(곡률 스케일 vs 45% floor)' },
    fit_for_a: { type: 'string', description: 'strong-target / weak / no — 3조건 종합' },
    experiment_sketch: { type: 'string', description: '이게 타겟이면: 무엇을 재계산, 어떤 figure가 flip을 보이나, 무엇이 필요한가' },
    notes: { type: 'string' },
  },
}

// ── Phase 1: Discover (멀티모달 병렬) ─────────────────────────────
phase('Discover')
const discovery = await parallel([
  // 1) 로컬 survey audit 채굴 — 이미 검토된 7편 + 그들의 곡률 계산법
  () => agent(
    `[목표] 9732 원고의 survey audit에서 latent-geometry 논문 후보를 추출한다.
[방법] Read/Grep으로 ${DIR}/note/submission/tmlr_paper.tex (Appendix G + Table 4), ${DIR}/note/submission/refs.bib (또는 ${DIR}/note/refs.bib), ${DIR}/note/TMLR_design.md 를 정독. 인용된 latent-geometry/curvature 논문 각각에 대해: 어떤 정성 결론을 내는가, 어떤 기하량(metric vs curvature/geodesic), 1차인가 2차인가, 곡률을 어떻게 계산하나(원고 audit이 뭐라 했나 — 0/7이 /eps^2 형성이라는데 그럼 각자 *대신* 뭘 쓰나).
${CRITERIA}
[경계] 로컬 파일에서 본 것만. 9732 무관 폴더(paper_refutation/tsfm_audit/higan_dev) 금지. 추측 금지.
[출력] 스키마(candidates).`,
    { label: 'discover:local-survey', phase: 'Discover', schema: CAND_SCHEMA }
  ),
  // 2) web — diffusion latent-space 곡률/geodesic 정성 주장
  () => agent(
    `[목표] diffusion/score-based 생성모델의 latent/data-space curvature·geodesic을 다루며 *정성 결론*을 내는 published 논문을 web에서 찾는다.
[방법] WebSearch로 "diffusion model latent space curvature", "score-based generative geometry geodesic", "diffusion semantic latent Riemannian curvature", "diffusion model trajectory curvature qualitative" 등 검색. 유망한 것은 WebFetch(arxiv abstract/페이지)로 확인. 각 후보에 곡률/geodesic이 2차인지, 정성 결론이 있는지, 코드 공개인지 기록.
${CRITERIA}
[경계] 실제 검색결과/페이지에서 본 것만. 환각 인용 금지 — ref에 url 또는 arxiv id 명시.
[출력] 스키마(candidates).`,
    { label: 'discover:web-diffusion', phase: 'Discover', schema: CAND_SCHEMA }
  ),
  // 3) web — GAN/VAE latent geometry 고전 (Shao/Arvanitidis/Kühnel/Chen/Yang)
  () => agent(
    `[목표] GAN/VAE deep generative model의 Riemannian latent geometry 고전·후속 논문 중 *2차 기하(곡률/geodesic/Christoffel)*로 정성 결론을 내는 것을 web에서 찾는다.
[방법] WebSearch로 "Riemannian geometry deep generative models Shao", "latent space oddity Arvanitidis", "geodesic VAE latent curvature", "GAN latent space curvature interpolation", "Christoffel symbols generative model latent" 등. WebFetch로 핵심 후보 확인 — 곡률/geodesic이 정성 결론(예: geodesic interpolation이 linear보다 자연스럽다, 특정 영역이 더 curved)에 쓰이는가, 코드 공개인가. **metric pullback(magnification)만 쓰는 1차 논문은 order=first로 표시(제외 대상).**
${CRITERIA}
[경계] 실제 페이지에서 본 것만. ref에 url/arxiv id.
[출력] 스키마(candidates).`,
    { label: 'discover:web-gan-vae', phase: 'Discover', schema: CAND_SCHEMA }
  ),
  // 4) web — 코드 공개 repo (재현성 우선 탐색)
  () => agent(
    `[목표] 생성모델 latent space에서 curvature/Christoffel/geodesic을 계산하는 *코드 공개* repo를 찾는다(재현성 우선 — bounded 재계산 가능한 타겟).
[방법] WebSearch로 "github generative model latent geodesic curvature code", "pytorch latent space Christoffel geodesic generative", "diffusion curvature github" 등. WebFetch로 repo README 확인 — pretrained generator로 곡률/geodesic을 계산하나, 정성 결론(논문)과 연결되나, 우리가 큰 학습 없이 돌릴 수 있나.
${CRITERIA}
[경계] 실제 repo/페이지에서 본 것만. ref에 github url.
[출력] 스키마(candidates).`,
    { label: 'discover:web-code', phase: 'Discover', schema: CAND_SCHEMA }
  ),
])

// 후보 통합 + 중복 제거(이름 기준, 코드)
const allCands = (discovery || []).filter(Boolean).flatMap(d => d.candidates || [])
const seen = new Set()
const uniqueCands = []
for (const c of allCands) {
  const k = (c.name || '').toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 40)
  if (k && !seen.has(k)) { seen.add(k); uniqueCands.push(c) }
}
log(`Discover: ${allCands.length} raw → ${uniqueCands.length} unique 후보`)

// ── Phase 2: Assess (후보별 3조건 평가, web 검증) ─────────────────
phase('Assess')
const assessed = await parallel(uniqueCands.map((c) => () =>
  agent(
    `[목표] 이 latent-geometry 후보가 9732 ⓐ 타겟으로 적합한지 3조건을 엄격 평가한다.
후보: ${JSON.stringify(c)}
[방법] 필요시 WebFetch/WebSearch로 원논문·repo를 직접 확인해 보강. 핵심 판정:
- cond1: 정성 결론이 실제 있나(있으면 무엇)?
- cond2: 그 결론이 *2차* 기하량에 의존하나? metric(1차)뿐이면 FAIL. 곡률·geodesic·Christoffel·2nd pushforward면 PASS.
- cond3: 코드/가중치/데이터로 우리가 bounded 재계산 가능한가? 작업량 추정.
- flip_plausibility: 곡률의 전형적 스케일 대비 FD floor(toy 8.6e-9, StyleGAN 45%)가 이 결론을 뒤집을 만큼 큰가?
${CRITERIA}
[경계] 확인한 것만. 모르면 "unknown"으로. 과장 금지 — TMLR을 살리고 싶은 편향에 저항하라(정직이 우선).
[출력] 스키마.`,
    { label: `assess:${(c.name || 'cand').slice(0, 24)}`, phase: 'Assess', schema: ASSESS_SCHEMA, effort: 'high' }
  )
))

// ── Phase 3: Rank + 완전성 비평 ──────────────────────────────────
phase('Rank')
const RANK_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['summary_md', 'ranked', 'top_pick', 'fallback_to_b', 'completeness_check', 'recommended_next'],
  properties: {
    summary_md: { type: 'string', description: '후보 × 3조건 × flip가능성 표' },
    ranked: { type: 'array', items: { type: 'string' }, description: 'fit 순 정렬(strong→no)' },
    top_pick: { type: 'string', description: '단일 추천 타겟 + 왜 + 구체 실험설계(무엇 재계산/어떤 figure가 flip/무엇 필요) — strong-target 없으면 "NONE"' },
    fallback_to_b: { type: 'string', description: 'strong-target가 없으면 왜 ⓑ(수치해석 venue)로 회귀해야 하는지 정직 판정. 있으면 "불필요".' },
    completeness_check: { type: 'string', description: '놓친 modality/검색각도/미확인 후보 — 다음 라운드에 더 팔 가치가 있나?' },
    recommended_next: { type: 'string', description: 'top_pick 있으면 그 실험 착수 단계(가중치/데이터/코드 획득→FD vs AD 재계산→flip figure). 없으면 ⓑ 재구성 골격.' },
  },
}
const rank = await agent(
  `[목표] 평가 결과로 9732 ⓐ 단일 타겟을 추천하거나, 없으면 ⓑ 회귀를 정직 판정한다.
[입력] 후보 평가: ${JSON.stringify((assessed || []).filter(Boolean))}
[요구] (1) 후보×3조건×flip 표(summary_md). (2) fit 순 ranked. (3) strong-target 1개를 top_pick으로 + 구체 실험설계. **3조건(정성결론∧2차의존∧재현가능)을 다 만족하고 flip_plausibility가 충분한 후보만 strong.** 없으면 top_pick="NONE". (4) strong 없으면 fallback_to_b를 정직하게(없는 consequence를 억지로 만들지 마라 — 플레이북 ${PLAYBOOK} §① 위반). (5) completeness_check: 검색 사각이 있어 한 라운드 더 팔 가치가 있는지. (6) recommended_next.
[경계] TMLR 편향 저항. 증거가 약하면 ⓑ. LLM-cadence·과장 금지.
[출력] 스키마.`,
  { phase: 'Rank', schema: RANK_SCHEMA, effort: 'high' }
)

return {
  raw_candidates: allCands.length,
  unique_candidates: uniqueCands.length,
  assessed: (assessed || []).filter(Boolean).length,
  result: rank,
}
