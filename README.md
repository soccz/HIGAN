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
- **분석 28종**: per-layer / 8×14 / disentanglement / local edit / encoder attention /
  random direction discovery / compositional / robustness / intermediate Grad-CAM /
  K-means taxonomy / CLIP zero-shot / ckpt evolution / ∂²I/∂α² / saliency morph /
  실제 LSUN 사진 등 — 모두 추가 학습 0개로 frozen generator + 기존 ckpt 위에서.
- **로컬 실행**: Colab 의존 제거, YAML config + 28-step CLI.

자세한 사용법과 모든 스크립트 인덱스는 [`higan_dev/README.md`](higan_dev/README.md). 30초 요약:

```bash
cd higan_dev
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps lpips==0.1.4 open_clip_torch
pip install pyyaml "numpy<2" pillow opencv-python tqdm matplotlib scipy \
            scikit-learn datasets ftfy regex wcwidth huggingface_hub safetensors timm

python scripts/01_download_assets.py            # HiGAN 자산
PYTHONPATH=. python scripts/02_invert_optim.py --self-test --steps 1000 --lr 0.1 --out out/inv_selftest
PYTHONPATH=. python scripts/03_train_encoder.py                  # 40k iter ≈ 1h 50m
PYTHONPATH=. python scripts/11_grad_saliency.py --num-samples 64 # JVP saliency (헤드라인)
PYTHONPATH=. python scripts/22_taxonomy.py                       # unsupervised attribute discovery
PYTHONPATH=. python scripts/24_clip_label_clusters.py            # CLIP zero-shot 라벨
```

## 결과 하이라이트

- **인버전**: optim 1000-step Adam이 self-test에서 loss 5.6 → 0.027 (99.5% ↓).
- **Saliency**: JVP 기반 gradient 버전이 각 침실의 실제 lamp / 창문 / 우드 프레임을 scene-specific 하게 pinpoint.
- **Disentanglement**: HiGAN의 8 boundary가 사실상 view + 표면 텍스처 두 클러스터로 갈림 (view만 직교 0.32, 나머지 7개 mutually 0.6–0.83 entangled).
- **Compositional**: 같은 layer cluster는 선형 합성 (carpet+wood corr 0.97), 교차는 비선형 간섭 (view+wood 0.55) — view의 곡률이 다른 attribute의 40배라는 ∂²I/∂α² 분석으로 직접 설명.
- **CLIP rediscovery**: 256개 random direction → K-means → CLIP zero-shot으로 cluster 2가 "a view through a window" 자동 식별. 라벨/분류기 0개로 HiGAN의 view boundary 재발견.
- **학습 동역학**: encoder의 saliency-vs-GT 상관이 1k → 40k에서 45배 향상 (recon MSE는 18%만), saliency가 reconstruction보다 학습에 민감.

## 라이선스

이 레포의 코드는 학술/실험용. genforce/higan의 가중치는 그쪽 라이선스를 따른다.
