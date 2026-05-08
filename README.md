# HIGAN

침실 GAN(StyleGAN bedroom256, [genforce/higan](https://github.com/genforce/higan))의 잠재공간을
**inversion + boundary 편집 + CAM-style 픽셀 saliency**로 해석하는 두 단계 프로젝트.

> 보고서: **<https://soccz.github.io/projects/higan/>**

## 두 가지 코드베이스

### v1 — Colab 노트북 (원본 프로토타입)

루트의 `.ipynb` 3종.

| 노트북 | 역할 |
| --- | --- |
| `HIGAN_encoder.ipynb` | 다중 손실(MSE / Perceptual / LPIPS / TV) 기반 최적화 인버전 |
| `HIGAN_encoder_지홍.ipynb` | 위 노트북과 동일 (이름만 다른 사본) |
| `HiGAN_PSP.ipynb` | pSp(FFHQ) 인코더로 잠재 추출 시도 |

손실함수
1. **MSE**  — 픽셀 차이 (색상·밝기 보정)
2. **Perceptual** — VGG16 feature 차이 (구조·패턴)
3. **LPIPS** — 시각적 유사성
4. **TV** — 매끄러움·노이즈 억제

알려진 한계: 가구 배치·구도는 잡히지만 색상·정확도는 부족.

### v2 — `higan_dev/` (.py 패키지, 본 보고서)

[`higan_dev/`](higan_dev/) 디렉토리에 로컬 GPU(RTX 3070 8 GB)용으로 다시 짠 모듈형 파이프라인.

핵심 차이점:
- **Differentiable generator wrapper**: v1의 `easy_synthesize`는 numpy uint8을 detach해서 반환 → autograd가 끊김.
  `higan_dev.generator.HiGANGenerator.synthesize(wp)`는 `G.net.synthesis`를 직접 호출해 gradient flow 보존.
- **도메인 특화 인코더**: pSp/FFHQ 대신 ResNet50 backbone + multi-scale neck + 14 layer-head를 합성 supervision으로 학습.
- **잠재 → 픽셀 saliency 두 가지 버전**:
  - forward perturbation (`cam/diff_map.py`): 분류기 없이 ±δ로 픽셀 차이 누적 — cheap, classifier-free, 평균 spatial 패턴.
  - backward gradient (`cam/grad_saliency.py`): 미분 가능 generator + `torch.func.jvp`로 ∂I/∂α를 정확히 계산 — 각 침실의 실제 램프/창문/우드 프레임을 pinpoint.
- **로컬 실행**: Colab 의존 제거, YAML config + CLI 8단계.

자세한 사용법은 [`higan_dev/README.md`](higan_dev/README.md) 참조. 30초 요약:

```bash
cd higan_dev
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps lpips==0.1.4
pip install pyyaml "numpy<2" pillow opencv-python tqdm matplotlib scipy

python scripts/01_download_assets.py            # HiGAN 자산 다운로드
PYTHONPATH=. python scripts/02_invert_optim.py --self-test --steps 1000 --lr 0.1 --out out/inv_selftest
PYTHONPATH=. python scripts/03_train_encoder.py # 20k iter ≈ 1h 47m on RTX 3070
PYTHONPATH=. python scripts/06_cam_analysis.py --num-samples 64 --out out/cam
PYTHONPATH=. python scripts/09_edit_real.py --ckpt out/encoder_train/ckpt/enc_020000.pt --out out/edit_real
```

## 결과 하이라이트

- **인버전**: optim 1000-step Adam이 self-test에서 loss 5.6 → 0.027 (99.5% ↓). v1이 풀지 못했던 정확 재구성 달성.
- **Saliency**: 8개 boundary 모두 의미상 정확한 위치에서 활성. forward perturbation 버전은 평균 패턴, JVP 기반 gradient 버전은 *각 침실의 실제 lamp / 창문 / 우드 프레임을 scene-specific 하게 pinpoint*. "잠재 방향이 픽셀 어디에 작용하는가"가 정량+spatial로 드러남.
- **인코더**: 1-pass inversion이 LPIPS 0.40 (under-fit 상태이나 편집 방향성은 보존됨). 추가 학습 시 0.20대 도달 예상.

## 라이선스

이 레포의 코드는 학술/실험용. genforce/higan의 가중치는 그쪽 라이선스를 따른다.
