"""aimslabs EURS Track 2 model: a small physics-guided restorer (PhysNet front end + half-resolution NAFNet U-Net,
0.42M params) and a MobileNetV3-Large segmentation student (about 0.33M params before export) behind one static input, plus a fixed
unsharp mask on the restored output. Copied from our training repository (restoration/eurs.py, seg/student.py,
restoration/export_onnx.py) with the branches the submitted model does not use removed; the traced graph is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

N_CLASSES = 5


def frame_stats(x):
    """Per-channel mean, std, max of the whole frame (B,3,H,W) -> (B,9). Biased std as a mean of squares
    (torch.std's N/(N-1) exports N=H*W as a constant that overflows FP16 in TensorRT)."""
    mu = x.mean((2, 3), keepdim=True)
    return torch.cat([mu.flatten(1), (x - mu).pow(2).mean((2, 3)).sqrt(), x.amax((2, 3))], 1)


def gauss(win=11, sigma=1.5):
    g = torch.exp(-((torch.arange(win) - win // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


class LayerNorm2d(nn.Module):
    """Channel LayerNorm, exported as ONNX LayerNormalization (opset 17)."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        x = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class NAFBlock(nn.Module):
    def __init__(self, c, film=0):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Conv2d(c, c, 1)
        self.c3 = nn.Conv2d(c, c, 1)
        self.c4 = nn.Conv2d(c, 2 * c, 1)
        self.c5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.film = nn.Linear(film, 2 * c) if film else None
        self.c = c

    def forward(self, x, z=None):
        c = self.c
        y = self.n1(x)
        if self.film is not None:
            f = self.film(z)[..., None, None]
            y = y * (1 + f[:, :c]) + f[:, c:]
        y = self.dw(self.c1(y))
        y = y[:, :c] * y[:, c:]
        y = y * self.sca(y.mean((2, 3), keepdim=True))
        x = x + self.c3(y) * self.beta
        y = self.c4(self.n2(x))
        return x + self.c5(y[:, :c] * y[:, c:]) * self.gamma


class GlobalHead(nn.Module):
    """Thumbnail (B,3,128,128) + frame stats -> global descriptor z."""

    def __init__(self, dim=64, width=32):
        super().__init__()
        ch = [3, width // 2, width, 2 * width, 2 * width]
        self.convs = nn.Sequential(*[m for i in range(4) for m in
                                     (nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), nn.GELU())])
        self.mlp = nn.Sequential(nn.Linear(2 * ch[-1] + 9, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, th, st):
        f = self.convs(th)
        return self.mlp(torch.cat([f.mean((2, 3)), f.amax((2, 3)), st], 1))


class LKA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.dwd = nn.Conv2d(c, c, 7, padding=9, groups=c, dilation=3)
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * self.pw(self.dwd(self.dw(x)))


class PhysNet(nn.Module):
    """Physics front end: global head -> posterior over the 200 (water type, level) cells -> posterior-weighted
    per-channel affine colour map (table fitted on the EURS training scenes) + a 1/8-res gain/offset veil map.
    `r` is unused by the BigNet restorer but kept so the checkpoint loads strictly."""

    def __init__(self, table, zdim=64, gw=32, vc=32, rc=16, down=8):
        super().__init__()
        self.down = down
        self.register_buffer("table", table.clone())
        self.g = GlobalHead(zdim, gw)
        self.cls = nn.Linear(zdim, 200)
        self.v_in = nn.Conv2d(6, vc, 3, padding=1)
        self.v_film = nn.Linear(zdim, 2 * vc)
        self.v_body = nn.Sequential(nn.GELU(), LKA(vc), nn.Conv2d(vc, vc, 3, padding=1), nn.GELU(),
                                    LKA(vc), nn.Conv2d(vc, 6, 1))
        self.r = nn.Sequential(nn.Conv2d(24, rc, 3, padding=1), nn.GELU(),
                               nn.Conv2d(rc, rc, 3, padding=1), nn.GELU(),
                               nn.Conv2d(rc, rc, 3, padding=1), nn.GELU(),
                               nn.Conv2d(rc, 12, 3, padding=1))

    def physics(self, x, th, st):
        z = self.g(th, st)
        logits = self.cls(z)
        with torch.autocast(x.device.type, enabled=False):
            p = logits.float().softmax(1)
            x, w = x.float(), p @ self.table.flatten(1)
            assert self.table.shape[-1] == 2, "submitted model uses the per-channel affine table"
            ab = w.view(-1, 3, 2)
            jt = x * ab[..., 0, None, None] + ab[..., 1, None, None]
        v = self.v_in(F.avg_pool2d(torch.cat([x, jt.clamp(0, 1)], 1), self.down))
        f = self.v_film(z)[..., None, None]
        vc = self.v_in.out_channels
        v = self.v_body(v * (1 + f[:, :vc]) + f[:, vc:])
        v = F.interpolate(v, scale_factor=self.down, mode="bilinear", align_corners=False)
        return jt * torch.exp(v[:, :3].clamp(-4, 4)) + v[:, 3:], z, logits


class BigNet(nn.Module):
    """PhysNet front end + NAFNet U-Net residual refiner (at 1/2 res via pixel (un)shuffle when half=True),
    conditioned on the global descriptor."""

    def __init__(self, table, width=32, enc=(1, 1, 2, 4), mid=6, dec=(1, 1, 1, 1), gw=64, half=False, grow=2.0):
        super().__init__()
        self.phys = PhysNet(table, gw=gw)
        self.half = half
        s = 4 if half else 1
        z = 64
        self.intro = nn.Conv2d(6 * s, width, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
        cs = [width]
        for n in enc:
            self.encs.append(nn.ModuleList([NAFBlock(cs[-1], z) for _ in range(n)]))
            cs.append(round(cs[-1] * grow))
            self.downs.append(nn.Conv2d(cs[-2], cs[-1], 2, 2))
        self.mid = nn.ModuleList([NAFBlock(cs[-1], z) for _ in range(mid)])
        for n, (c_in, c) in zip(dec, zip(cs[:0:-1], cs[-2::-1])):
            self.ups.append(nn.Sequential(nn.Conv2d(c_in, 4 * c, 1), nn.PixelShuffle(2)))
            self.decs.append(nn.ModuleList([NAFBlock(c, z) for _ in range(n)]))
        self.ending = nn.Conv2d(width, 3 * s, 3, padding=1)
        self.pad = 2 ** len(enc) * (2 if half else 1)

    def forward(self, x, th, st):
        j0, z, logits = self.phys.physics(x, th, st)
        H, W = x.shape[-2:]
        assert H % self.pad == 0 and W % self.pad == 0
        inp = torch.cat([x, j0.clamp(0, 1)], 1)
        h = self.intro(F.pixel_unshuffle(inp, 2) if self.half else inp)
        skips = []
        for blocks, down in zip(self.encs, self.downs):
            for b in blocks:
                h = b(h, z)
            skips.append(h)
            h = down(h)
        for b in self.mid:
            h = b(h, z)
        for up, blocks, s in zip(self.ups, self.decs, reversed(skips)):
            h = up(h) + s
            for b in blocks:
                h = b(h, z)
        r = self.ending(h)
        r = F.pixel_shuffle(r, 2) if self.half else r
        return (j0 + r).clamp(0, 1), j0, logits


def t2a_restorer(table):
    """The 0.42M-param Track 2 restorer ("t2a" in our training code)."""
    return BigNet(table, width=32, enc=(1, 2, 2), mid=2, dec=(1, 1, 1), gw=32, half=True, grow=1.25)


class Student(nn.Module):
    """Segmentation student, arch "l": MobileNetV3-Large blocks 0-10 (1/8 and 1/16 features) + gated context and a
    1/8 fusion head. Input RGB 0..1, logits at the input size."""

    def __init__(self):
        super().__init__()
        f = torchvision.models.mobilenet_v3_large(weights=None).features
        self.low, self.high, cl, ch, width = f[:7], f[7:11], 40, 80, 96
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.cbr = nn.Sequential(nn.Conv2d(ch, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, width, 1), nn.Sigmoid())
        self.mid = nn.Sequential(nn.Conv2d(width + cl, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
                                 nn.ReLU())
        self.cls = nn.Conv2d(width, N_CLASSES, 1)

    def forward(self, x):
        H, W = int(x.shape[-2]), int(x.shape[-1])
        assert H % 32 == 0 and W % 32 == 0
        xn = (x - self.mean) / self.std
        lo = self.low(xn)
        hi = self.high(lo)
        y = self.cbr(hi) * self.gate(hi)
        y = F.interpolate(y, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
        y = self.mid(torch.cat([y, lo], 1))
        y = self.cls(y)
        return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)


class Track2(nn.Module):
    """x (1,3,H,W) in 0..1 -> (restored_image, seg_logits).
    restored: restorer output + unsharp mask y + k (y - gauss_sigma(y)) (fixed depthwise 11x11 Conv, reflect pad).
    seg: student on the degraded input, averaged with its hflip, mixed with a half-scale pass on a
    saturation-boosted view (weight `half`)."""

    def __init__(self, restorer, segmenter, half, half_sat, unsharp, unsharp_sigma):
        super().__init__()
        self.restorer, self.seg, self.half, self.half_sat, self.unsharp = restorer, segmenter, half, half_sat, unsharp
        self.blur = nn.Conv2d(3, 3, 11, groups=3, bias=False)
        self.blur.weight.data.copy_(gauss(sigma=unsharp_sigma).expand(3, 1, 11, 11))

    def forward(self, x):
        th = F.adaptive_avg_pool2d(x, 128)  # == area resize; exports as AveragePool (384x768 is divisible)
        st = frame_stats(x)
        restored, _, _ = self.restorer(x, th, st)
        blur = self.blur(F.pad(restored, (5,) * 4, mode="reflect"))
        restored = (restored + self.unsharp * (restored - blur)).clamp(0, 1)
        s = self.seg(x)
        s = (s + self.seg(x.flip(-1)).flip(-1)) * 0.5
        g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        xh = (g + self.half_sat * (x - g)).clamp(0, 1)
        sh = self.half * self.seg(F.avg_pool2d(xh, 2))
        s = (1 - self.half) * s + F.interpolate(sh, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return restored, s
