# Method description

Team: aimslabs   Track(s): this repository = Track 2 (we also entered Track 1, separate repository)
Reported testing-phase submission: Track 2 #19380 (aimslabs_t2_final = t2_usms05k20_74, user #1995, composite 0.430)

## Approach
One ONNX graph, static input (1, 3, 384, 768) RGB in [0, 1], opset 17, 0.737 M parameters, **11.380 GFLOPs**
(the organizer checker prints 11.166: it gives 0 FLOPs to the unsharp-mask Conv because that Conv follows a reflect
Pad whose output shape ONNX shape inference leaves unknown; the true total is still under the 12 GFLOPs cap).

```
x ─┬─> restorer (0.42 M) ─> unsharp mask y + 2.0 (y - G_0.5(y)) ─> restored_image
   ├─> student(x) ────────────────┐
   ├─> student(hflip x), flip ────┴─ mean ─┐
   └─> saturation x1.5 -> AvgPool 2x2 -> student -> bilinear x2 ─ 0.67 / 0.33 mix ─> seg_logits
```

- **Restorer** (`t2a_restorer` in `src/t2model.py`): a physics front end reads a 128x128 thumbnail plus 9 frame
  statistics, predicts a softmax over the 200 (water type, depth level) cells and applies the softmax-weighted
  per-cell affine colour table (fitted on the 20 EURS training scenes) at full resolution, plus a 1/8-resolution
  large-kernel-attention gain/offset map for depth-varying veil. A NAFNet U-Net at 1/2 resolution (pixel
  unshuffle, channels 32-40-50-62, FiLM on the global descriptor) adds a residual. Nothing is read from file names.
- **Unsharp mask**: fixed depthwise 11x11 Gaussian (sigma 0.5), reflect padding, amount 2.0, clamped to [0, 1]
  (the evaluator upsamples our 384x768 output; this restores the sharpness term).
- **Segmentation student** (`Student` in `src/t2model.py`, 0.33 M): torchvision MobileNetV3-Large blocks 0-10 with an
  LR-ASPP-style gated head fused at 1/8 resolution. Run three times with shared weights: the input, its horizontal
  flip, and a half-scale pass on a saturation-boosted view. Per-class logit offsets (0, -0.375, 0.375, -1.0, -0.125)
  are added to the classifier bias in `export_onnx.py`.

## Training data
- **EURS**: all 20 training scenes (all water types and levels). Architectures and recipes were compared on
  scene-grouped folds of these scenes.
- **EURS dev degraded images (6 scenes, no GT)**: the restorer's last fine-tuning stage uses the dev degraded frames
  with targets made by a Track 1 restorer (A) checkpoint from the same frame (30% of sampled examples; organizer
  ruling 17 Oct 2026: dev is training data, disclose).
- **Testing-phase images**: no testing-phase pixels were used for training, pseudo-labelling or adaptation.
  Testing-phase leaderboard scores of our own submissions selected the student seed, the logit-offset scale and the
  unsharp-mask sigma and amount. These settings are global and fixed for every image.
- **External data** (all public; links and licences also in the factsheet):
  LSUI (Peng et al., TIP 2023; github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement; no
  explicit licence) raw photos, re-degraded with our colour model fitted on EURS, for the restorer (50% of sampled examples, 40% in
  the last stage);
  unlabelled photos for distilling the student: LSUI, UIIS10K (Li et al. 2025; huggingface.co/datasets/LiamLian0727/UIIS10K;
  Apache-2.0), AquaOV255 (Li et al. 2025, arXiv 2511.07923; huggingface.co/datasets/kkk2026/UOVSBench; dataset
  card: Apache-2.0; article: CC BY 4.0) and Coralscapes V1/V2 frames (Sauder et al. 2025 / 2026;
  huggingface.co/datasets/EPFL-ECEO/coralscapes, huggingface.co/datasets/josauder/coralscapesV2; Apache-2.0). Their
  original annotations are not used: the photos are colour-matched and re-degraded, and the student learns the
  teacher's logits and features on them, with supervised losses only on labelled AquaOV255 cut-outs pasted into crops.
  Pool size: 2,577 LSUI, 20,723 AquaOV255, 9,638 UIIS10K, 2,075 Coralscapes and 2,433 CoralscapesV2 photos (37,446).
- **Pretrained models**: MobileNetV3-Large ImageNet-1k (torchvision `IMAGENET1K_V1`, BSD-3) initialises the student.
  The distillation teacher is our Track 1 segmenter (DINOv3 ViT-L/16 LVD-1689M, Meta, DINOv3 License, fine-tuned
  on the 20 EURS scenes); it is used only during training and is not in the ONNX. The restorer's dev targets come
  from our Restormer-based Track 1 restorer, initialised from the public `real_denoising.pth` weights (MIT;
  https://github.com/swz30/Restormer); it only generates training targets and is not in the ONNX.
- Training: restorer 3 stages (30k + 20k + 5k iterations, L1 + 0.2 SSIM + 0.3 j0 + 0.05 cell classification);
  student 40k iterations, batch 8, crop 512 on the squashed 384x768 geometry, weighted CE + Dice on GT +
  KL(teacher || student) at T = 2 + feature distillation, student in FP32 with an FP16 teacher forward, seed 3.

## Anything non-standard at inference
Everything is inside the ONNX graph (hflip and half-scale segmentation passes, logit offsets, unsharp mask).
No external pre- or post-processing.

## Reproduction notes
`python export_onnx.py weights out.onnx` (or `main(model_dir, out_onnx)`) rebuilds the ONNX on CPU in a few seconds.
Inside the organizers' base image:

```bash
docker build -t eurs-base docker/
mkdir -p out
docker run --rm --network none -v "$PWD":/pkg:ro -v "$PWD/out":/out \
  eurs-base python /pkg/export_onnx.py /pkg/weights /out/model.onnx
```

- With torch 2.11.0 (what we used for the upload) the export is byte-identical to `onnx/submitted_model.onnx`
  (SHA-256 `52a79e046d91a658c6054eb889dfb72fda71cc1bd1c6fe9180f32489692e8e9e`).
- With the base image's torch 2.4.0 + torchvision 0.19.0, `repro_check.py --export-onnx` reports the same graph and
  weights (header differs); ONNX Runtime outputs are bit-identical to the submitted file on 9 inputs. This does not establish identical TensorRT FP16 results on the
  Jetson; that needs the organizers' rerun.
- Weights (committed): `weights/restorer_t2kd_all.pt` (restorer), `weights/seg_student_l_s3.pt` (student).
- Training code is not part of this hand-in (export only); the factsheet describes the training recipe.
