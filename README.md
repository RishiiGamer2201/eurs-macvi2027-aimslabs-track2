# aimslabs: MaCVi 2027 EURS, Track 2 reproduction

Export code for our Track 2 testing-phase entry (see `repro.yaml` for the submission ID, `METHOD.md` for the method,
data and licences). Made from the organizers' template `CeviKle/EURS-MaCVi2027-repro-template`.

```
export_onnx.py              main(model_dir, out_onnx): rebuilds the submitted ONNX from weights/
src/t2model.py              restorer and segmentation student
weights/                    restorer_t2kd_all.pt, seg_student_l_s3.pt (committed)
onnx/submitted_model.onnx   exact file uploaded to the leaderboard
docker/Dockerfile           the organizers' base image (torch 2.4.0), unchanged
repro_check.py              the organizers' checker, unchanged
```

## Run

```bash
python export_onnx.py weights out.onnx        # CPU, a few seconds
python repro_check.py . --export-onnx         # rebuilds and compares with onnx/submitted_model.onnx
```

Inside the organizers' base image:

```bash
docker build -t eurs-base docker/
mkdir -p out
docker run --rm --network none -v "$PWD":/pkg:ro -v "$PWD/out":/out \n  eurs-base python /pkg/export_onnx.py /pkg/weights /out/model.onnx
```

ONNX: static input (1, 3, 384, 768) RGB in [0, 1], outputs `restored_image` and `seg_logits`, opset 17, 0.737 M parameters,
11.38 GFLOPs. No extra packages beyond the base image (`requirements.txt` is empty).

## License

Our code: MIT (`LICENSE`). The student uses torchvision's MobileNetV3-Large ImageNet weights (BSD-3) as initialisation.
