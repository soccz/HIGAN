# Related work survey — 59 papers across 12 categories

> Compiled 2026-05-09 by Explore agent. **Status**: draft v0, dates and venues to verify before BibTeX export.
> Strategy: this file is the human-readable index; final BibTeX export will be done into `paper/references.bib` after pruning.

---

## 1. GAN interpretation & saliency (14)

### Foundational dissection & understanding

- **[Bau2019GANDissection]** Bau, D.; Zhou, B.; Khosla, A.; Oliva, A.; Torralba, A. *GAN Dissection: Visualizing and Understanding Generative Adversarial Networks.* ICLR 2019.
  - Unit-level, object-level, scene-level analysis via segmentation-based dissection. Quantifies causal effects via ablation.
  - **Relation**: closest prior on spatial attribution. Our method differs by being classifier-free and gradient-based on the generator alone.

- **[Shen2020InterFaceGAN]** Shen, Y.; Gu, J.; Tang, X.; Zhou, B. *Interpreting the Latent Space of GANs for Semantic Face Editing.* CVPR 2020.
  - Linear semantic subspaces via boundary-seeking hyperplanes on FFHQ. Source of FFHQ boundaries we'll use as input.
  - **Relation**: provides boundaries; we provide a geometric account of their interaction.

- **[Harkonen2020GANSpace]** Härkönen, E.; Hertzmann, A.; Lehtinen, J.; Paris, S. *GANSpace: Discovering Interpretable GAN Controls.* NeurIPS 2020.
  - PCA on activation tensors → unsupervised directions.
  - **Relation**: alternative discovery method; we compare on rediscovery of hand-curated boundaries.

- **[Shen2021SeFa]** Shen, Y.; Zhou, B. *Closed-Form Factorization of Latent Semantics in GANs.* CVPR 2021.
  - Eigendecomposition of affine layer → global semantic directions.
  - **Relation**: algebraic closed-form; ours is sample-based local geometric.

- **[Wu2021StyleSpace]** Wu, Z.; Lischinski, D.; Shechtman, E. *StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation.* CVPR 2021.
  - Style space (channels) more disentangled than W; per-channel manipulations.
  - **Relation**: per-channel; we generalise to arbitrary tangent directions and cluster.

- **[Yang2020HiGAN]** Yang, C.; Shen, Y.; Zhou, B.; Kuo, C.-C. J. *Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis.* IJCV 2020.
  - HiGAN: hierarchical semantic organisation across layers. Source of bedroom boundaries.
  - **Relation**: parent method; provides bedroom boundaries we benchmark against.

### StyleGAN architecture & latent mapping

- **[Karras2019StyleGAN]** Karras, T.; Laine, S.; Aila, T. *A Style-Based Generator Architecture for GANs.* CVPR 2019.
- **[Karras2018ProGAN]** Karras, T.; Aila, T.; Laine, S.; Lehtinen, J. *Progressive Growing of GANs.* ICLR 2018.
- **[Karras2020StyleGAN2]** Karras, T.; Laine, S.; Aila, T.; et al. *Analyzing and Improving the Image Quality of StyleGAN.* CVPR 2020.
  - **Relation**: backbone generators used in our experiments.

### Advanced editing & compositionality

- **[Ling2021EditGAN]** Ling, H.; Kreis, K.; Li, D.; Kim, S. W.; Torralba, A.; Fidler, S. *EditGAN: High-Precision Semantic Image Editing.* NeurIPS 2021.
  - Joint image + segmentation modelling; reusable edit vectors.
  - **Relation**: orthogonal approach (segmentation supervision); useful comparison for edit precision.

- **[Patashnik2021StyleCLIP]** Patashnik, O.; Wu, Z.; Shechtman, E.; Cohen-Or, D.; Lischinski, D. *StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery.* ICCV 2021.
  - **Relation**: shares CLIP as text supervision; we use CLIP only for cluster labelling, not editing.

- **[Goetschalckx2019GANalyze]** Goetschalckx, L.; Andonian, A.; Oliva, A.; Isola, P. *GANalyze: Toward Visual Definitions of Cognitive Image Properties.* ICCV 2019.
  - Learned latent-space trajectories as vector fields.
  - **Relation**: closest framing to our "attribute as vector field" view.

- **[Jahanian2020Steerability]** Jahanian, A.; Chai, L.; Isola, P. *On the "Steerability" of Generative Adversarial Networks.* ICLR 2020.
  - Self-supervised trajectories for camera/colour transforms; linear vs non-linear steerability.
  - **Relation**: directly relevant — our ∂²I/∂α² quantifies their "non-linear" case.

### Generator inversion

- **[Abdal2019Image2StyleGAN]** Abdal, R.; Qin, Y.; Wonka, P. *Image2StyleGAN.* ICCV 2019.
- **[Richardson2021pSp]** Richardson, E.; Alaluf, Y.; Patashnik, O.; et al. *Encoding in Style (pSp).* CVPR 2021.
- **[Tov2021e4e]** Tov, O.; Alaluf, Y.; Patashnik, O. *Designing an Encoder for StyleGAN Image Manipulation (e4e).* SIGGRAPH 2021.
- **[Alaluf2021ReStyle]** Alaluf, Y.; Patashnik, O.; Cohen-Or, D. *ReStyle.* ICCV 2021.
- **[Roich2022PTI]** Roich, D.; Shaham, T. S.; Bermano, A. H.; Cohen-Or, D. *Pivotal Tuning for Latent-based Editing of Real Images.* ACM TOG 2022.
  - **Relation**: inversion baselines; our encoder (§04) compares against this family.

---

## 2. Classical saliency & attribution (11)

### Visual attribution

- **[Zhou2016CAM]** Zhou, B.; Khosla, A.; Lapedriza, A.; Oliva, A.; Torralba, A. *Learning Deep Features for Discriminative Localization.* CVPR 2016.
- **[Selvaraju2017GradCAM]** Selvaraju, R. R.; Cogswell, M.; Das, A.; et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- **[Chattopadhyay2018GradCAMpp]** Chattopadhyay, A.; Phan, A.; Dasgupta, A.; Sarkar, S. *Grad-CAM++.* WACV 2018.
- **[Petsiuk2018RISE]** Petsiuk, V.; Das, A.; Saenko, K. *RISE.* BMVC 2018.
  - **Relation**: classifier-based attribution. Our paper positions JVP-pushforward as the *generative* analogue.

### Integrated attribution & Shapley

- **[Sundararajan2017IntegratedGradients]** Sundararajan, M.; Taly, A.; Yan, Q. *Axiomatic Attribution for Deep Networks.* ICML 2017.
- **[Lundberg2017SHAP]** Lundberg, S. M.; Lee, S.-I. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
- **[Ribeiro2016LIME]** Ribeiro, M. T.; Singh, S.; Guestrin, C. *LIME.* KDD 2016.
  - **Relation**: axiomatic frameworks; our pushforward satisfies trivially the sensitivity axiom.

### Information-theoretic

- **[Schulz2021IBA]** Schulz, K.; et al. *Information Bottleneck Attribution.* ICLR 2020 (verify venue).
- **[Bach2016LRP]** Bach, S.; Binder, A.; Montavon, G.; et al. *Layer-Wise Relevance Propagation.*
  - **Relation**: information-flow attribution; orthogonal axis.

---

## 3. Differential geometry & Riemannian methods in NN (6)

- **[Amari1998NaturalGradient]** Amari, S. *Natural Gradient Works Efficiently in Learning.* Neural Computation 10(2), 1998.
  - **Relation**: Fisher metric foundational; our generator manifold is a different but related geometry.

- **[Amari2019Fisher]** Amari, S.; Karakida, R. *Fisher Information and Natural Gradient Learning of Random Deep Networks.* ICML 2019.

- **[Bonnabel2013RiemannianSGD]** Bonnabel, S. *Stochastic Gradient Descent on Riemannian Manifolds.* JMLR 2013.

- **[Kasai2019RiemannianAdaptive]** Kasai, H.; Sato, H.; Lederer, J.; Mishra, B. *Riemannian Adaptive Stochastic Gradient Algorithms on Matrix Manifolds.* ICML 2019.

- **[Bronstein2021GeometricDL]** Bronstein, M. M.; Bruna, J.; Cohen, T.; Veličković, P. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* ICLR 2021 keynote / book.
  - **Relation**: meta-framework. We instantiate GDL principles for *trained* generator manifolds.

- **[Cohen2016GroupEquivariant]** Cohen, T.; Welling, M. *Group Equivariant Convolutional Networks.* ICML 2016.

---

## 4. Forward-mode AD & second-order methods (5)

- **[Pearlmutter1994HVP]** Pearlmutter, B. A. *Fast Exact Multiplication by the Hessian.* Neural Computation 6(1), 1994.
  - **Foundational**. The "Pearlmutter trick" of forward-over-reverse for HVP. Modern JAX/PyTorch implementations descend from this paper. **Cite prominently.**

- **[Martens2015KFAC]** Martens, J.; Grosse, R. *Optimizing Neural Networks with Kronecker-factored Approximate Curvature.* ICML 2015.
- **[Martens2011HessianFree]** Martens, J.; Sutskever, I. *Training Deep and Recurrent Networks with Hessian-Free Optimization.* ICML 2011.

- **[PyTorch2023functorch]** PyTorch. *Jacobians, Hessians, HVP, VHP, and more: composing function transforms.* functorch docs, 2023.
  - **Relation**: our implementation framework.

- **[JAX2023Autodiff]** JAX Team. *The Autodiff Cookbook: forward, reverse, mixed-mode.* JAX docs.

---

## 5. Disentanglement & manifold learning (8)

- **[Higgins2017BetaVAE]** Higgins, I.; et al. *β-VAE: Learning Basic Visual Concepts.* ICLR 2017.
- **[Kim2018FactorVAE]** Kim, H.; Mnih, A. *Disentangling by Factorising.* NeurIPS 2018.
- **[Eastwood2020DisentanglementMetrics]** Eastwood, C.; Williams, C. K. *A Framework for the Quantitative Evaluation of Disentangled Representations.* ICLR 2018 (verify year).
  - **Relation**: quantitative disentanglement metrics — we provide a geometric source for them via stratification.

- **[Chan2024ManifoldTraining]** Chan, S. H.; et al. *The Training Process of Many Deep Networks Explores the Same Low-Dimensional Manifold.* PNAS 2024.
- **[Li2024DeepManifold]** Li, M.; Soriano, J.; Li, Y. *Deep Manifold Part 1: Anatomy of Neural Network Manifold.* arXiv:2409.17592, 2024.

- **[Zhu2016ManipulationOnManifold]** Zhu, J.-Y.; Krähenbühl, P.; Shechtman, E.; Efros, A. A. *Generative Visual Manipulation on the Natural Image Manifold.* ECCV 2016.
  - **Relation**: earliest "image manifold" framing for GANs.

---

## 6. Vector fields & Lie groups (2)

- **[Finzi2021LieConv]** Finzi, M.; Welling, M.; et al. *Generalizing CNNs for Equivariance to Lie Groups on Arbitrary Continuous Data.* ICLR 2021.
- **[Weiler2019SteerableFilters]** Weiler, M.; Cesa, G.; Welling, M.; Cohen, T. *Learning Steerable Filters for Rotation Equivariant CNNs.* CVPR 2019.

---

## 7. GAN stability & training (3)

- **[Miyato2018SpectralNorm]** Miyato, T.; Kataoka, T.; Koyama, M.; Yoshida, Y. *Spectral Normalization for GANs.* ICLR 2018.
- **[Gulrajani2017WGANGP]** Gulrajani, I.; Ahmed, F.; Arjovsky, M.; Dumoulin, V.; Courville, A. C. *Improved Training of Wasserstein GANs.* NeurIPS 2017.
- **[Salimans2016ImprovedGANTraining]** Salimans, T.; Goodfellow, I.; Zaremba, W.; et al. *Improved Techniques for Training GANs.* NeurIPS 2016.

---

## 8. Modern architectures & attention (2)

- **[Dosovitskiy2021ViT]** Dosovitskiy, A.; et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- **[Chefer2021TransformerInterpretability]** Chefer, H.; Benaim, S.; Cohen-Or, D.; Irani, M. *Transformer Interpretability Beyond Attention Visualization.* CVPR 2021.

---

## 9. Feature visualization & dissection (3)

- **[Yosinski2016FeatureViz]** Yosinski, J.; Clune, J.; Bengio, Y.; Lipson, H. *Multifaceted Feature Visualization.* arXiv:1602.03616, 2016.
- **[Yosinski2015DeepVisualization]** Yosinski, J.; et al. *Understanding Neural Networks Through Deep Visualization.* ICML DL Workshop 2015.
- **[Zhou2018AblationStudy]** Zhou, B.; Sun, Y.; Neumann, U.; Torralba, A. *Revisiting the Importance of Individual Units in CNNs via Ablation.* arXiv:1806.02891, 2018.

---

## 10. Adversarial perspectives (2)

- **[Goodfellow2015AdversarialExamples]** Goodfellow, I. J.; Shlens, J.; Szegedy, C. *Explaining and Harnessing Adversarial Examples.* ICLR 2015.
- **[Madry2018AdversarialTraining]** Madry, A.; Makelov, A.; Schmidt, L.; Tsipras, D.; Vladu, A. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.

---

## 11. Information theory (1)

- **[Tishby2020InformationBottleneck]** Tishby, N.; Zaslavsky, N. *Deep Learning and the Information Bottleneck Principle.* arXiv:1503.02406 / 2015 ITW.

---

## 12. Prototype networks / case-based (1)

- **[Chen2019PrototypeNetworks]** Chen, C.; Li, O.; Tao, D.; Barnett, A.; Rudin, C.; Su, J. K. *Deep Learning for Case-Based Reasoning through Prototypes.* AAAI 2019 (verify venue from arXiv 1710.04806).

---

## Strategic positioning

Our paper sits squarely between:

| neighbour | what they do | what we add |
| --- | --- | --- |
| GAN Dissection / SeFa / GANSpace | discrete unit/direction-level attribution | continuous tangent-space + Hessian framework |
| Grad-CAM family | classifier-side attribution | generator-side attribution, classifier-free |
| Pearlmutter / functorch HVP | forward-mode for optimisation | forward-mode for *interpretation* of generator manifolds |
| Geometric Deep Learning | inductive biases on inputs | analysis of *learned* manifolds |
| InterFaceGAN / HiGAN boundaries | provide directions to interpret | we explain *why* compositional editing fails for some pairs |

## Action items before BibTeX export

- [ ] Verify all 2019–2024 venue dates (some marked "verify").
- [ ] Add 5–8 more recent papers (2023–2024 GAN inversion / interpretability survey).
- [ ] Prune to ~45 BibTeX entries for the actual paper (CVPR has reference page limit).
- [ ] Cross-check `Shen2021SeFa` and `Shen2020InterFaceGAN` are the same Shen Yujun (yes) — disambiguate keys.

## Notable gaps the survey identified

1. Very few papers explicitly on *curvature* in GAN latent spaces — our second-order analysis is genuinely novel.
2. Limited prior on Lie-bracket composition of attribute directions.
3. Random-tangent-direction clustering for manifold stratification appears novel.
4. Pushforward / pullback formalism in GAN context underexplored.

These four gaps are exactly the slots we plan to fill.
