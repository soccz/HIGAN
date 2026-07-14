# TMLR 리라이트 킥오프 — 9732 유한차분 (신규 세션 진입점)

> 새(클린) 세션이 이 폴더에서 가장 먼저 읽는 문서. 옛 원고를 쓴 컨텍스트로 돌아가지 마라(그 컨텍스트가 리젝 프레임을 만들었다). 무편향 시작. (근거: [[feedback_blind_objective_review]])
>
> ⚠️ 이 폴더엔 9732 외 다른 라인(`paper_refutation/`, `tsfm_audit/`, `higan_dev/`, `paper/`)이 섞여 있다. **9732 증거는 `note/` + `logs/track*.log` + `note/submission/evidence/` 에만** 있다. 나머지는 건드리지 마라.

## 0. 진입 절차
1. (필요시) `/clear` 후 이 폴더에서 시작.
2. **"TMLR 논문용이야"** → 글로벌 스킬 `tmlr-paper` 로드.
   - 네거티브 플레이북(필독): `/home/soccz/22tb/.claude-packs/tmlr/guide/NEGATIVE_AUDIT_PLAYBOOK.md`
3. **쓰기 전에** ②(evidence inventory)부터. 옛 원고(`note/submission/tmlr_paper.tex`)는 *숫자 소스*로만.

## 1. 이 논문의 진단 (검증된 사유)
- **죽인 것:** ①자기부정(`"most methods avoid the regime by construction"`, `"we show no downstream task win ... left to future work"`) → at-risk 독자 자멸 ②메타 hedge 도배(`"we do not claim any published result is wrong"`) ③노트로 읽힘(교과서 수치해석 √ε_mach + "textbook" nested AD) ④marquee가 n=6 seed + near-zero outlier에 의존.
- **견고한 자산(리드로 써야 할 것):** Proposition 1(√ε_mach 비복구성 증명) + StyleGAN step sweep + fp32/fp64 통제 + toy 해석해 대조(`note/toy_fd_groundtruth.json`). 이건 deterministic·airtight.

## 2. ⚠️ 가장 먼저 할 결정 — VENUE 갈림길 (쓰기 전)
합격 네거티브 7편은 **예외 없이** 발견을 *바뀌는 결정*으로 닫았다(플레이북 §①). 9732는 이게 비어서 죽었다. 그래서:

> **핵심 질문: FD vs exact-JVP의 차이가 *어떤 다운스트림 ML 결론을 바꾸는* 결과가 있나, 또는 만들 수 있나?**
> (예: 곡률 기반 clustering/ordering/정성 비교가 FD에선 A, exact에선 B로 뒤집힘)

- **있다/만들 수 있다 → TMLR.** 그 결과를 리드 처방으로, 자기부정 전삭, deterministic 결과를 본문 중심으로.
- **없다/데이터 부담으로 불가 → 수치해석/autodiff venue**(SIAM SIMODS·ACM TOMS·differentiable-programming 워크숍). 거기선 "exact higher-order AD > finite differences"가 다운스트림 win 없이 그 자체로 1급 기여.

evidence inventory(②)가 `logs/track*`에서 다운스트림 단서(track17_intrinsic·track19_dino·track10_per_layer 등)를 우선 탐색해 이 갈림길에 답을 준다.

## 3. 결과(증거) 위치
- 옛 원고: `note/submission/tmlr_paper.tex` (+ working: `note/tmlr_main.tex`)
- 해석해 대조: `note/toy_fd_groundtruth.json` / `.py`
- 실험 로그: `logs/track*.log` (FD 관련: track18_fd_validation, track1b_sd_n64, track11_walltime, track13_resolution; 다운스트림 후보: track17_intrinsic_{bedroom,ffhq}, track19_dino, track10_per_layer)
- 제출 증거: `note/submission/evidence/`

## 4. 실행 순서
```
# 1) evidence inventory + VENUE 갈림길 판정
Workflow({scriptPath: "/home/soccz/22tb/study/HIGAN/evidence_inventory.workflow.js"})
# 2) venue 확정 후 → 플레이북 §③ 구조로 NEW manuscript 신규 작성 (옛 것 편집 금지)
#    deterministic 결과 리드, 네거티브를 바뀌는 결정으로 닫기
# 3) 게이트: tmlr-desk-editor → tmlr-criteria-auditor → tmlr-style-surgeon → paper-cold-reader
```
게이트 에이전트는 이 폴더 `.claude/agents/`에 이미 부착됨.
