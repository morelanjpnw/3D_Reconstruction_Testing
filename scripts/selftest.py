"""GPU self-test: triggers gsplat's one-time CUDA compile (several minutes) and runs a tiny forward+backward."""
import time

import torch
from gsplat.rendering import rasterization

print("torch", torch.__version__, "| cuda", torch.version.cuda, "| device", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0))
t0 = time.time()
n = 2000
d = "cuda"
means = (torch.randn(n, 3, device=d) * 0.5 + torch.tensor([0, 0, 3.0], device=d)).requires_grad_()
quats = torch.nn.functional.normalize(torch.randn(n, 4, device=d), dim=-1)
scales = torch.full((n, 3), 0.03, device=d)
opac = torch.full((n,), 0.5, device=d)
colors = torch.rand(n, 3, device=d)
K = torch.tensor([[200.0, 0, 128], [0, 200.0, 128], [0, 0, 1]], device=d)[None]
out, _, _ = rasterization(means, quats, scales, opac, colors, torch.eye(4, device=d)[None], K, 256, 256)
out.mean().backward()
torch.cuda.synchronize()
print(f"gsplat OK (render {tuple(out.shape)}, grad finite: {bool(torch.isfinite(means.grad).all())}) in {time.time() - t0:.0f}s")
