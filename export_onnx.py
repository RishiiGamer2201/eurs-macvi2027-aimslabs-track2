"""Track 2: regenerate our submitted ONNX (aimslabs, t2_usms05k20_74) from the weights in weights/.

    main(model_dir, out_onnx)          # organizer entry point
    python export_onnx.py weights out.onnx

Static input (1, 3, 384, 768) RGB in [0, 1]; outputs restored_image (1,3,384,768) and seg_logits (1,5,384,768).
Opset 17, legacy TorchScript exporter, no simplifier. The only graph edit after torch.onnx.export points consumers of
Identity-aliased shared weights (the student runs three times) straight at the initializers, so the organizer's
FLOP counter sees every Conv.
"""
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.t2model import Student, Track2, t2a_restorer  # noqa: E402

H, W = 384, 768
SEG_BIAS = [0.0, -0.375, 0.375, -1.0, -0.125]  # per-class logit offsets (bg, coral, fish, elasmobranch, other)
HALF, HALF_SAT = 0.33, 1.5                     # half-scale seg pass: weight, saturation of its input
UNSHARP, UNSHARP_SIGMA = 2.0, 0.5              # restored + k (restored - gauss_sigma(restored))


def build(model_dir):
    sd = torch.load(os.path.join(model_dir, "restorer_t2kd_all.pt"), map_location="cpu", weights_only=True)
    rest = t2a_restorer(sd["phys.table"])
    rest.load_state_dict(sd)
    seg = Student()
    seg.load_state_dict(torch.load(os.path.join(model_dir, "seg_student_l_s3.pt"), map_location="cpu",
                                   weights_only=True))
    seg.cls.bias.data += torch.tensor(SEG_BIAS)
    return Track2(rest, seg, HALF, HALF_SAT, UNSHARP, UNSHARP_SIGMA).eval()


def main(model_dir, out_onnx):
    import onnx
    torch.manual_seed(0)
    model = build(model_dir)
    os.makedirs(os.path.dirname(out_onnx) or ".", exist_ok=True)
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model, (torch.rand(1, 3, H, W),), out_onnx, input_names=["input"],
                      output_names=["restored_image", "seg_logits"], opset_version=17, **kw)
    m = onnx.load(out_onnx)
    inits = {i.name for i in m.graph.initializer}
    alias = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Identity" and n.input[0] in inits}
    for n in m.graph.node:
        for k, v in enumerate(n.input):
            if v in alias:
                n.input[k] = alias[v]
    keep = [n for n in m.graph.node if not (n.op_type == "Identity" and n.output[0] in alias)]
    del m.graph.node[:]
    m.graph.node.extend(keep)
    onnx.save_model(m, out_onnx, save_as_external_data=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weights", sys.argv[2] if len(sys.argv) > 2 else "out/model.onnx")
