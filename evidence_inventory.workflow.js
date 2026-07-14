export const meta = {
  name: 'fd9732-evidence-inventory',
  description: '9732 유한차분: 리젝 원고 주장을 note/+logs/track* 증거와 대조해 load-bearing/filler 분류 + 가장 중요한 VENUE 갈림길(다운스트림 consequence 결과가 있나/만들 수 있나 → TMLR vs 수치해석 venue) 판정. 결과=데이터, 옛 narrative 폐기. 9732 외 라인(paper_refutation/tsfm_audit/higan_dev/paper)은 제외.',
  phases: [
    { title: 'Map', detail: '원고 주장 + note/·logs/track* 결과 색인 (9732 관련만)' },
    { title: 'Verify', detail: '주장별 증거 대조 + 다운스트림-consequence 단서 탐색' },
    { title: 'Synthesize', detail: '인벤토리 + VENUE 갈림길 판정 + 단일 confident 발견' },
  ],
}

const DIR = '/home/soccz/22tb/study/HIGAN'
const PLAYBOOK = '/home/soccz/22tb/.claude-packs/tmlr/guide/NEGATIVE_AUDIT_PLAYBOOK.md'

const MAP_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['claims', 'result_index', 'downstream_candidates'],
  properties: {
    claims: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['id', 'text', 'section', 'asserted_numbers', 'kind'],
      properties: {
        id: { type: 'string' }, text: { type: 'string' }, section: { type: 'string' },
        asserted_numbers: { type: 'string' },
        kind: { type: 'string', description: 'deterministic(증명/sweep/fp대조) | statistical(n=6 등) | audit-claim | downstream' },
        is_headline: { type: 'boolean' },
      } } },
    result_index: { type: 'array', items: { type: 'string' }, description: 'note/ + logs/track* 중 9732 관련 결과파일 + 한 줄 설명 (paper_refutation/tsfm_audit/higan_dev 제외)' },
    downstream_candidates: { type: 'array', items: { type: 'string' }, description: 'FD vs exact 차이가 다운스트림 결론을 바꿀 가능성이 있는 결과/로그 (track17_intrinsic·track19_dino·track10_per_layer 등). 없으면 빈 배열.' },
  },
}
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['claim_id', 'claim', 'evidence_path', 'number_matches', 'robustness', 'verdict', 'changes_a_decision', 'note'],
  properties: {
    claim_id: { type: 'string' }, claim: { type: 'string' },
    evidence_path: { type: 'string', description: '근거 파일 경로(못 찾으면 NONE)' },
    number_matches: { type: 'string', description: 'match/mismatch/not-found' },
    robustness: { type: 'string', description: 'deterministic·airtight 인지, 아니면 약한 통계(n=6·outlier 의존)인지' },
    changes_a_decision: { type: 'string', description: '핵심: 이 결과가 *어떤 다운스트림 ML 결론/결정을 바꾸나*? (yes+무엇 / no / potential+어떻게)' },
    verdict: { type: 'string', enum: ['load-bearing', 'filler', 'unsupported'] },
    note: { type: 'string' },
  },
}

phase('Map')
const map = await agent(
  `[목표] 9732 유한차분 리젝 원고의 주장을 목록화하고 note/+logs/track* 결과를 색인한다. **9732 관련만** — paper_refutation/·tsfm_audit/·higan_dev/·paper/ 는 무시.
[방법] (1) ${DIR}/note/submission/tmlr_paper.tex (+ ${DIR}/note/tmlr_main.tex) 를 Read로 정독, 핵심 주장을 한 문장씩 + 수치 + kind(deterministic/statistical/audit/downstream). (2) Bash로 ${DIR}/logs/track*.log 와 ${DIR}/note/ , ${DIR}/note/submission/evidence/ 를 나열하고 각 track 로그가 무엇을 측정했는지 grep으로 한 줄 요약. (3) FD vs exact 차이가 다운스트림 결론을 바꿀 단서가 있는 로그를 downstream_candidates로 분리(track17_intrinsic, track19_dino, track10_per_layer 우선 확인).
[경계] 추출만. 옛 narrative 프레이밍 불신. 9732 무관 폴더 접근 금지.
[출력] 스키마.`,
  { phase: 'Map', schema: MAP_SCHEMA }
)

phase('Verify')
const verifs = await parallel((map?.claims || []).map((c) => () =>
  agent(
    `[목표] 9732의 이 주장이 증거로 뒷받침되는지 + *다운스트림 결정을 바꾸는지* 검증한다.
주장 [${c.id} / ${c.section} / ${c.kind}]: "${c.text}"  (수치: ${c.asserted_numbers})
[방법] ${DIR}/logs/track*.log, ${DIR}/note/, ${DIR}/note/submission/evidence/ 에서 Grep/Read로 근거 확인. deterministic(증명·sweep·fp대조)인지 약한 통계(n=6·outlier)인지 판정. **핵심: 이 결과가 어떤 다운스트림 ML 결론을 바꾸나? (changes_a_decision)**
[기준(플레이북 ${PLAYBOOK} §①⑥)] 다운스트림 결정을 바꾸면 load-bearing. deterministic·airtight면 load-bearing(리드감). "측정만 틀림·downstream win 없음"이면 filler. 증거 약하면 unsupported.
[경계] 실제 파일에서 본 것만. 9732 무관 폴더 금지.
[출력] 스키마.`,
    { label: `verify:${c.id}`, phase: 'Verify', schema: VERIFY_SCHEMA }
  )
))

phase('Synthesize')
const SYN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['inventory_md', 'venue_verdict', 'downstream_consequence_status', 'single_confident_finding', 'load_bearing', 'filler_to_cut', 'recommended_next'],
  properties: {
    inventory_md: { type: 'string', description: 'claim↔evidence 인벤토리 표' },
    downstream_consequence_status: { type: 'string', description: 'FD vs exact가 다운스트림 결론을 바꾸는 결과가 (이미 있음 / 단서있음·소량작업으로 가능 / 없음·큰 실험필요) 중 무엇인지 + 근거' },
    venue_verdict: { type: 'string', description: '위에 근거해 단일 권고: TMLR(다운스트림 consequence 확보/가능 시) vs 수치해석 venue(SIAM/TOMS/diff-prog, consequence 불가 시). 하나로 명확히.' },
    single_confident_finding: { type: 'string', description: '리라이트 앵커 한 문장(자기부정·hedge 0). venue에 맞게: TMLR이면 바뀌는 결정 포함, 수치해석이면 instrument 우위.' },
    load_bearing: { type: 'array', items: { type: 'string' } },
    filler_to_cut: { type: 'array', items: { type: 'string' } },
    recommended_next: { type: 'string', description: 'venue가 TMLR이면 어떤 다운스트림 결과를 어떤 로그/실험으로 만들지; 수치해석이면 어떤 재구성인지. 플레이북 §③ 구조 골격 포함.' },
  },
}
const syn = await agent(
  `[목표] 검증 결과로 9732 evidence inventory + VENUE 갈림길 판정을 낸다.
[입력] 검증: ${JSON.stringify((verifs || []).filter(Boolean))}
[요구] (1) 인벤토리 표. (2) downstream_consequence_status를 증거로 판정. (3) 그에 근거해 venue_verdict를 **하나로** 명확히(TMLR vs 수치해석). (4) deterministic 자산을 리드로 한 단일 confident 발견. (5) filler 명시. (6) recommended_next(다음 실험 또는 재구성 + 새 골격).
[경계] 옛 narrative 복원 금지. 자기부정·과hedge·LLM-cadence 범하지 마라. 증거가 다운스트림을 안 바꾸면 정직히 수치해석 venue 권고.
[출력] 스키마.`,
  { phase: 'Synthesize', schema: SYN_SCHEMA }
)

return { claims: (map?.claims || []).length, verified: (verifs || []).filter(Boolean).length, downstream_candidates: map?.downstream_candidates || [], inventory: syn }
