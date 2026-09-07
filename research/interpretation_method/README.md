# 개입 반응의 중첩과 비선형 상호작용 분석

연구의 기준 문서는 [INTERACTION.md](../../INTERACTION.md)다.
목표는 생성모델 내부 표현에 대한 해석 방법론이다.
[방향 설계](DIRECTION.md), [수학 정의](DESIGN.md),
[첫 실제 모델 결과](PILOT_REPORT.md)를 함께 읽는다.

**상태: 2026-09-07 연구 종료·결과 공개.**
[짧은 보고서](REPORT.md) · [결과 소개 페이지](https://soccz.github.io/projects/higan-interaction/)

## 공개 자료와 빠른 검증

GitHub에는 측정 코드, 고정 프로토콜, 행 단위 JSONL, 사전에 저장한 예측,
학습 모델·보정 임계값, 결과표와 감사 기록을 포함한다.
대용량 원출력 텐서·잠재 텐서·실행 시 소스 복사본은 로컬에 보존하고 Git에서는 제외했다.
모델 가중치와 업스트림 코드는 별도로 준비한다.

Python 3.12 환경에서 아래 명령으로 공개 v2 자료의 해시, 예측 기록, 학습·보정 및
기준점 bootstrap 결과를 CPU로 재검산한다. 이 검사는 원래 GPU 출력의 재생성을 대신하지 않는다.

```bash
python3 -m pip install 'numpy<2' 'torch==2.2.2'
python3 -m unittest discover -s research/interpretation_method -p 'test_*.py' -v
python3 research/interpretation_method/verify_public.py
```

원래 환경은 Python 3.12.9·torch 2.2.2+cu121이다. GPU 재현은
[의존성 설치 순서](../../higan_dev/requirements.txt)를 따르고 저장소 루트에서 자산을 준비한다.

```bash
python3 higan_dev/scripts/01_download_assets.py --data-dir higan_dev/data
git -C higan_dev/data/higan_repo checkout 2088e23cfd1e2ab86ce6eb4edf2abe1701cbb9d9
```

업스트림의 가중치 배포 링크와 라이선스는 [genforce/higan](https://github.com/genforce/higan)을 따른다.
가중치·8개 방향·소스의 기대 SHA-256은 각 실행의 `manifest.json`에 있다.
`explain.py`는 공개된 보정 파일과 위 자산으로 실행하며 입력·소스 해시가 다르면 중단한다.

아래의 원본 실행 폴더는 이미 공개 결과를 담고 있다. **전체 실험을 다시 실행할 때는 모든
단계를 새 폴더명으로 연결**해야 한다. 예를 들어 `selective_v2_*`를 `repeat_selective_v2_*`로 바꾼다.
전체 `verify_selective.py` 감사에는 새 실행이 생성한 텐서와 소스 복사본이 필요하다.
원본 manifest의 절대 경로는 실행 당시 기록이며, 다른 컴퓨터의 경로로 수정해 배포하지 않는다.
원본 공개 자료 검산에는 경로 이동을 지원하는 `verify_public.py`를 사용한다.

## 현재 구현

- `measure.py`: 부호 보존 유한 반응, 혼합 반응, 상쇄, 제곱거리의 중첩/비선형 항.
  제곱 관측에 대한 정확한 유한 분해와 매끄러운 점의 일반 관측 연쇄법칙을 구분한다.
- `test_measure.py`: 실제 구현의 해석해·반례 검사 11개.
- `protocol_bedroom_v1.json`: 실행 전에 고정한 24개 기준점·8방향·28쌍·3크기·4부호 조합.
- `run_bedroom.py`: 기존 가중치로 사전 검증과 직접 관측 파일럿을 실행한다.
  출력은 원래 생성기의 unclamped fp32 값, 측정 연산은 fp64다.
- `verify_pilot.py`: 누락·중복·비유한 값·소스 및 입력 해시·분해 항등식을 검사한다.
  기존 조작 함수와 원래 synthesis forward로 저장 사례를 독립 재계산할 수 있다.

## 실행

저장소 루트에서 실행한다. 현재 환경은 Python 3.12.9, torch 2.2.2+cu121이다.
GPU 실행은 CUDA에 접근할 수 있는 실행 환경이 필요하다.

```bash
python3 -m unittest discover -s research/interpretation_method -p test_measure.py -v
python3 research/interpretation_method/verify_pilot.py research/interpretation_method/runs/bedroom_v1_pilot
python3 research/interpretation_method/verify_pilot.py research/interpretation_method/runs/bedroom_v1_pilot --rerender-seed 92016
```

새 실행에는 아직 없는 출력 디렉터리를 지정한다. 기존 결과는 덮어쓰지 않는다.

```bash
python3 research/interpretation_method/run_bedroom.py --mode preflight --protocol research/interpretation_method/protocol_bedroom_v1.json --out research/interpretation_method/runs/repeat_preflight
python3 research/interpretation_method/run_bedroom.py --mode pilot --protocol research/interpretation_method/protocol_bedroom_v1.json --preflight research/interpretation_method/runs/repeat_preflight --out research/interpretation_method/runs/repeat_pilot
```

실행은 사전 검증 당시 프로토콜·측정 소스·가중치/방향 해시가 동일할 때만 진행한다.
소스 스냅샷, 각 기준점의 잠재값, 사전에 지정한 한 방향 쌍의 부호 보존 텐서를
로컬 결과에 보관한다. 이 대용량 재현 자료는 Git 추적에서 제외하며 로컬에는 유지한다.

## 해석 범위

이 파일럿은 직접 관측한 반응의 특성 분석이다. 8,064개 행은 독립 표본 8,064개가
아니며 독립적으로 뽑은 기준점은 24개다. 의미 정확도·미관측 반응 예측·새 방법의
우월성을 측정한 실험은 아직 아니다. 관측기의 제곱 변환은 해석해가 있는 대조 조건이다.

## 미관측 공동 반응 예측 실험

`protocol_prediction_v1.json`은 기존 개발 기준점 8개와 새 평가 기준점 16개를 분리한다.
`prediction.py`는 목표의 단독 반응과 작은 공동 개입만 입력으로 사용한다.
`run_prediction.py`는 개발 계수를 고정한 뒤 각 기준점의 예측 파일을 저장하고,
그 이후 목표 공동 출력을 생성한다. `verify_prediction.py`는 이 순서와 해시,
학습 자료 분리, 전체 격자, 저장 텐서, 기준점 단위 bootstrap을 감사한다.
현재 결과와 판정은 루트 `INTERACTION.md`에서 갱신한다.

```bash
python3 -m unittest discover -s research/interpretation_method -p 'test_*.py' -v
python3 research/interpretation_method/run_prediction.py --mode preflight --protocol research/interpretation_method/protocol_prediction_v1.json --out research/interpretation_method/runs/prediction_v1_preflight
python3 research/interpretation_method/run_prediction.py --mode train --protocol research/interpretation_method/protocol_prediction_v1.json --preflight research/interpretation_method/runs/prediction_v1_preflight --out research/interpretation_method/runs/prediction_v1_training
python3 research/interpretation_method/run_prediction.py --mode evaluate --protocol research/interpretation_method/protocol_prediction_v1.json --training research/interpretation_method/runs/prediction_v1_training --out research/interpretation_method/runs/prediction_v1_evaluation
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 research/interpretation_method/verify_prediction.py research/interpretation_method/runs/prediction_v1_evaluation --rerender-seed 93016
```

재실행에는 아직 없는 출력 디렉터리를 지정한다. 이 비교는 유한 혼합 반응의
외삽과 국소 미분의 실제 효용을 확인하는 단계이며, 외삽 식 자체의 신규성을 주장하지 않는다.

v1 감사는 저장한 예측값과의 완전 일치를 검사하므로 실행기와 같은 CPU 스레드 수 4를 쓴다.
기본 6스레드에서는 최대 `5.55e-17`의 합산 순서 차이가 발생했다. 측정 소스를 변경하지 않고
4스레드로 재검사해 전체 텐서 오차 0을 확인했다. 후속 버전은 계산 환경을 감사 진입점에서
명시적으로 설정하고 수치 허용 기준도 구분해야 한다.

`analyze_prediction.py`는 평가 후의 진단이다. 저장된 세 노름에서 probe와 목표 혼합
반응의 내적을 복원해, 양의 계수만 바꿀 때 가능한 최저 오차를 구한다.
정답을 사용한 한계 분석이며 예측 성능이나 사전에 고정한 성공 판정에 넣지 않는다.

```bash
python3 research/interpretation_method/analyze_prediction.py research/interpretation_method/runs/prediction_v1_evaluation
```

## 두 관측과 보류 v2

학습 24개, 채택 기준 보정 16개, 평가 64개 기준점을 분리한다.
목표 단독 반응과 0.01·0.03 공동 관측으로 0.06·0.08·0.1의 가산 설명 위험을 예측한다.
`selective.py`가 위험 회귀·독립 보정·같은 채택률 비교·기준점 신뢰구간을 계산한다.
`run_selective.py`는 모든 목표 예측을 저장한 뒤 목표 공동 출력을 생성한다.
`verify_selective.py`는 이전 단계까지 재귀적으로 감사하고 원래 생성기로 사례를 재계산한다.

```bash
python3 -m unittest discover -s research/interpretation_method -p 'test_*.py' -v
python3 research/interpretation_method/run_selective.py --mode preflight --protocol research/interpretation_method/protocol_selective_v2.json --out research/interpretation_method/runs/selective_v2_preflight
python3 research/interpretation_method/run_selective.py --mode train --protocol research/interpretation_method/protocol_selective_v2.json --previous research/interpretation_method/runs/selective_v2_preflight --out research/interpretation_method/runs/selective_v2_training
python3 research/interpretation_method/run_selective.py --mode calibrate --protocol research/interpretation_method/protocol_selective_v2.json --previous research/interpretation_method/runs/selective_v2_training --out research/interpretation_method/runs/selective_v2_calibration
python3 research/interpretation_method/run_selective.py --mode evaluate --protocol research/interpretation_method/protocol_selective_v2.json --previous research/interpretation_method/runs/selective_v2_calibration --out research/interpretation_method/runs/selective_v2_evaluation
python3 research/interpretation_method/verify_selective.py research/interpretation_method/runs/selective_v2_evaluation --rerender-seed 96064
```

v2 감사는 진입 시 4스레드를 설정하고 `1e-12` 상대·절대 수치 기준으로 재계산을 비교한다.
프로토콜·소스·입력·결과 파일·예측 저장 해시는 완전 일치해야 한다.
경험적 bootstrap 구간과 보류 임계값을 분포에 무관한 위험 보장으로 보고하지 않는다.
일괄 50% 채택은 각 기준점·목표 크기 안에서의 순위 비교다.
단일 입력의 채택 여부는 별도 보정에서 고정한 임계값으로 평가한다.

보조 분석과 정해진 경계의 판정:

```bash
python3 research/interpretation_method/analyze_selective.py research/interpretation_method/runs/selective_v2_evaluation
python3 research/interpretation_method/decide_selective.py research/interpretation_method/runs/selective_v2_evaluation
```

판정 경계가 불명확하면 미리 예약한 확인 기준점 64개를 `--mode confirm`으로 실행한다.
`--previous`는 같은 `selective_v2_calibration`을 사용한다. 평가 자료로 재학습하거나
임계값을 바꾸지 않는다. 확인 실험도 불명확하면 해당 효과를 미확정으로 기록한다.

v2는 평가 64개·21,504건과 독립 재계산까지 완료했다. 지정한 경계가 구분돼 추가 확인은
실행하지 않았다. 기본 방법은 가까운 관측+위험 회귀+보류다. 상세 수치와 주장 범위는
루트 `INTERACTION.md`의 현재 결론과 v2 평가 결과를 따른다.

## 검증한 기본 방법 실행

`explain.py`는 동결한 학습 모델·보정 임계값으로 `adopt_additive` 또는 `abstain`을 출력한다.
작은 공동 개입은 0.03 하나만 사용하며 목표 0.06·0.08·0.1의 공동 출력을 조회하지 않는다.
8방향 전체 격자에서 177회 호출로 336개 판단을 만든다. 실제 모델에서 seed 96064의
336개 점수와 판단이 검증 실험과 차이 0으로 일치했다. 이 사례는 구현 재현용이다.

```bash
python3 research/interpretation_method/explain.py --seed 96064 --out research/interpretation_method/runs/near_explainer_96064
```

재실행에는 새 출력 디렉터리를 사용한다. 범위 밖 모델·방향·크기에 대한 보정을 제공하지 않으며,
현재 결정은 개별 사례의 인증이 아닌 검증된 분포와 범위의 경험적 선택 규칙이다.
