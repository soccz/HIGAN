# HIGAN — a research line that ended honestly

> StyleGAN 잠재공간 해석에서 출발해, 2차 미분기하의 한계를 **검증된 negative**로 확정하고,
> 그 폐허의 감사에서 살아남은 아이디어 하나를 논문으로 완성하기까지의 전 과정 기록.
> 이 레포는 성공만 남기지 않는다 — 실패한 가설, 발동한 킬게이트, 적대검증이 잡아낸
> artifact까지 전부 커밋돼 있다.

**📖 읽기 좋은 형태의 기록:**
- [HIGAN v2 보고서](https://soccz.github.io/projects/higan/) — 28종 해석 분석 (JVP saliency · disentanglement · CLIP 재발견)
- [일반화 여정](https://soccz.github.io/projects/higan-generalization/) — 2차 곡률 라인의 전체 실험 기록 (95 versions, 정직한 회고)
- [폐기에서 논문까지](https://soccz.github.io/projects/higan-to-paper/) — 이 폴더의 감사에서 시작해 논문이 되기까지 7일

---

## 이 레포의 아크

```
2024-12  LumTerior (부트캠프 서비스: StyleGAN bedroom + Grad-CAM 조명 배치)
            │  한계 자기명시: "사용자 이미지 → latent 인코더가 없다"
2026-05  v2 재점화: 미분가능 generator 래퍼 + 인코더 + 28종 해석 분석
            │  발견: view 방향의 곡률이 texture류의 수십 배
2026-05  "곡률 = 편집 위험 신호" 가설 → 사전등록 컨트롤 캠페인 95 versions
            │  판정: controller가 church에서 random에 패배 → 라인 폐기
2026-06  피벗: "FD가 2차 기하의 부호를 뒤집는다" (TMLR 제출 → desk-reject)
            │  리라이트: exact composed-JVP instrument + step-selection trilemma
            │  워크플로우 6종의 consequence 사냥 → "다운스트림 결과 없음" 확정
2026-06-22  ABANDONED — 실패가 아니라 검증된 negative로 종결
2026-07-07  이 폴더의 전수 감사 → tsfm_audit의 banked 관찰 하나가 부활
2026-07-14  "Data-Starved Baselines Inflate the Measured Advantage of
             Time-Series Foundation Models on ETT" (8pp) 완성 → arXiv
```

## 검증된 negative (이 레포의 핵심 산출물)

FD(finite-difference) 기반 2차 generator 곡률에 대해, fixed-seed 증거로 확정한 것:

| 결과 | 수치 | 증거 |
|---|---|---|
| FD /δ² magnitude 오차 floor | fp32 ≥45% (fp64도 47.12% — 정밀도 무관) | `note/submission/evidence/` |
| step-selection trilemma | magnitude/bias/rank 최적 step이 3.0/2.0/0.2로 분리 | 〃 |
| **그러나 rank는 보존** | exact 대비 Spearman 0.815–0.963 | 〃 |
| → 다운스트림 consequence 부재 | ordering flip 0, gate flip은 rescale artifact | 〃 |

즉: FD 곡률은 크기로는 틀리지만 순위로는 맞아서, 순위 기반 응용에서는 **아무 결정도
뒤집히지 않는다**. 이것이 이 라인이 논문이 되지 못한 이유이고, 그 사실 자체가 기록 가치다.

부산물로 남은 도구: **exact composed-JVP 2차 곡률 계측기**
(`higan_dev/higan_dev/generator.py` — toy 해석해 대비 상대오차 2.5e-17, 비트동일 결정성).

## 레포 구조

```
higan_dev/          v2 해석 파이프라인 (scripts/01–28: 분석 28종, 29–33: FD-vs-exact 라인)
paper/              곡률-제어 라인: 사전등록 프로토콜 95개 + 컨트롤 캠페인 (743 runs, 0 fail)
note/               TMLR 라인 정본: 설계 문서, 원고 2판, 제출 evidence (fixed-seed JSON)
paper_refutation/   자기반박 시도의 하루 기록 (leakage audit 방법론 포함)
tsfm_audit/         TSFM 오염 감사 곁가지 — 후속 논문의 씨앗이 된 banked 관찰
*.workflow.mjs      데스크리젝 후 재작성을 오케스트레이션한 에이전트 워크플로우 6종
MEMORY.md           실험 연속성 로그 (종결 기록 포함)
```

v2 파이프라인의 설치·실행 인덱스는 [README_v2_pipeline.md](README_v2_pipeline.md).

## 방법론적으로 남긴 것

이 레포가 논문보다 오래 쓰일 수 있는 부분:

- **사전등록 하네스** — 잠긴 JSON 프로토콜에 판정 규칙을 실험 전에 박고, 클린 트리에서
  실행하고, 결과에 프로토콜 해시·커밋을 기록 (`paper/experiments/protocols/`)
- **적대검증 패턴** — 무맥락 cold 감사, deflationary-prior consequence 사냥,
  "작성자가 자기 산출물을 승인하지 않는다" 게이트 (루트 워크플로우 6종)
- **artifact 판별 기법** — global-rescale 후 irreducible flip 판정, outcome-coupled
  baseline의 label-leakage 정량 감사 (`paper_refutation/leakage_audit.py`)

## 후속

이 폴더의 감사에서 부활한 라인은 별도로 완성됐다:
**Data-Starved Baselines Inflate the Measured Advantage of Time-Series Foundation Models on ETT**
— TSFM의 zero-shot 우위 중 일부가 baseline 학습예산의 측정 artifact임을 개입 실험으로 실증.
(arXiv 링크 공개 시 여기 게시. 여정 전체는 [폐기에서 논문까지](https://soccz.github.io/projects/higan-to-paper/).)

## 라이선스

코드는 학술/실험용. `genforce/higan` 가중치는 원 저장소 라이선스를 따른다.
