"""Core library: COLMAP IO, flythrough paths, gsplat training / rendering / evaluation.

Frame manifest format (list of dicts): file, c2w (4x4, OpenCV convention), K (3x3, for
nominal size w x h), w, h, split ('train'|'test'), name. If the image on disk has a
different size than (w, h) (e.g. upscaled frames), K is scaled automatically at load time.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

DEV = "cuda"
SH_C0 = 0.28209479177387814


# ----------------------------------------------------------------------------- IO
def load_colmap(sparse_dir: Path, images_dir: Path, test_every: int = 8):
    """Read an undistorted (PINHOLE) COLMAP model. Returns (frames, points_xyz, points_rgb)."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    frames = []
    for i, im in enumerate(sorted(rec.images.values(), key=lambda x: x.name)):
        cam = rec.cameras[im.camera_id]
        cfw = im.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        w2c = np.eye(4)
        w2c[:3, :4] = cfw.matrix()
        p = list(cam.params)
        if len(p) == 3:  # SIMPLE_PINHOLE
            p = [p[0], p[0], p[1], p[2]]
        K = [[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]]
        frames.append(dict(file=str(images_dir / im.name), name=im.name,
                           c2w=np.linalg.inv(w2c).tolist(), K=K, w=int(cam.width), h=int(cam.height),
                           split="test" if i % test_every == 0 else "train"))
    pts = list(rec.points3D.values())
    xyz = np.array([q.xyz for q in pts], dtype=np.float32)
    rgb = np.array([q.color for q in pts], dtype=np.float32) / 255.0
    return frames, xyz, rgb


def save_manifest(path: Path, frames, xyz=None, rgb=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frames))
    if xyz is not None:
        np.savez(path.with_suffix(".points.npz"), xyz=xyz, rgb=rgb)


def load_manifest(path: Path):
    return json.loads(Path(path).read_text())


def load_points(path: Path):
    d = np.load(Path(path).with_suffix(".points.npz"))
    return d["xyz"], d["rgb"]


def read_image(f: dict):
    img = cv2.cvtColor(cv2.imread(f["file"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    K = np.array(f["K"], dtype=np.float32).copy()
    K[0] *= w / f["w"]
    K[1] *= h / f["h"]
    return img, K


def scene_radius(frames) -> float:
    c = np.array([np.array(f["c2w"])[:3, 3] for f in frames])
    return float(np.linalg.norm(c - c.mean(0), axis=1).max()) * 1.1


# ----------------------------------------------------------------------------- paths
def make_path(train_frames, n_out: int, lateral=0.0, vertical=0.0, yaw_deg=0.0, pitch_deg=0.0,
              smooth_sigma=2.0):
    """Smooth flythrough through the (ordered) training poses; optional local offset.

    lateral / vertical are fractions of the camera-orbit radius (vertical > 0 = down, OpenCV).
    Returns frames (no 'file' yet) with c2w, K, w, h copied from the first training frame.
    """
    fr = sorted(train_frames, key=lambda f: f["name"])
    c2w = np.array([f["c2w"] for f in fr])
    n = len(fr)
    t_in = np.arange(n)
    pos = gaussian_filter1d(c2w[:, :3, 3], smooth_sigma, axis=0, mode="nearest")
    rots = Rotation.from_matrix(c2w[:, :3, :3])
    t_out = np.linspace(0, n - 1, n_out)
    P = CubicSpline(t_in, pos)(t_out)
    R = Slerp(t_in, rots)(t_out)
    # smooth the rotation path by slerping through low-passed rotvecs
    rv = gaussian_filter1d(R.as_rotvec(), smooth_sigma, axis=0, mode="nearest")
    R = Rotation.from_rotvec(rv)
    scale = scene_radius(fr) / 1.1
    local = Rotation.from_euler("yx", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    out = []
    for i in range(n_out):
        M = np.eye(4)
        M[:3, :3] = R[i].as_matrix() @ local
        M[:3, 3] = P[i] + R[i].as_matrix() @ np.array([lateral * scale, vertical * scale, 0.0])
        out.append(dict(c2w=M.tolist(), K=fr[0]["K"], w=fr[0]["w"], h=fr[0]["h"],
                        name=f"{i:05d}.png", split="train"))
    return out


# ----------------------------------------------------------------------------- model
def _init_params(xyz, rgb, scene_scale, seed):
    g = torch.Generator().manual_seed(seed)
    n = len(xyz)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=4)
    d = np.clip(d[:, 1:].mean(1), 1e-6, None)
    means = torch.from_numpy(xyz).float()
    scales = torch.log(torch.from_numpy(d).float()).unsqueeze(1).repeat(1, 3)
    quats = torch.rand(n, 4, generator=g) * 2 - 1
    opac = torch.logit(torch.full((n,), 0.5))
    sh = torch.zeros(n, 16, 3)
    sh[:, 0] = (torch.from_numpy(rgb).float() - 0.5) / SH_C0
    return dict(means=means, scales=scales, quats=quats, opacities=opac,
                sh0=sh[:, :1].contiguous(), shN=sh[:, 1:].contiguous())


def _render(params, c2w, K, W, H, sh_degree):
    from gsplat.rendering import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], 1)
    view = torch.linalg.inv(c2w)[None]
    out, alpha, info = rasterization(
        params["means"], F_norm(params["quats"]), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, view, K[None], W, H,
        sh_degree=sh_degree, packed=False)
    return out[0].clamp(0, 1), info


def F_norm(q):
    return torch.nn.functional.normalize(q, dim=-1)


def train(frames, xyz, rgb, out_dir: Path, steps=7000, cap=500_000, seed=0, real_weight=None,
          log=print):
    """Train a splat on frames (split=='train'). real_weight: probability of drawing a frame
    flagged is_real (anchoring); None = uniform sampling."""
    from gsplat.strategy import MCMCStrategy
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = [f for f in frames if f["split"] == "train"]
    scale = scene_radius(tr)
    p0 = _init_params(xyz, rgb, scale, seed)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(DEV)) for k, v in p0.items()})
    lrs = dict(means=1.6e-4 * scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    strat = MCMCStrategy(cap_max=cap, refine_stop_iter=int(steps * 0.85), verbose=False)
    strat.check_sanity(params, opts)
    state = strat.initialize_state()

    cache = {}

    def get(i):
        if i not in cache:
            img, K = read_image(tr[i])
            cache[i] = (torch.from_numpy(img), torch.from_numpy(K))
            if len(cache) > 400:
                cache.pop(next(iter(cache)))
        return cache[i]

    real_idx = [i for i, f in enumerate(tr) if f.get("is_real")]
    syn_idx = [i for i, f in enumerate(tr) if not f.get("is_real")]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(steps):
        if real_weight is not None and real_idx and syn_idx:
            pool = real_idx if np.random.rand() < real_weight else syn_idx
            i = int(np.random.choice(pool))
        else:
            i = int(np.random.randint(len(tr)))
        img, K = get(i)
        gt = img.to(DEV).float() / 255.0
        H, W = gt.shape[:2]
        c2w = torch.tensor(tr[i]["c2w"], dtype=torch.float32, device=DEV)
        pred, info = _render(params, c2w, K.to(DEV), W, H, min(step // 1000, 3))
        strat.step_pre_backward(params, opts, state, step, info)
        l1 = (pred - gt).abs().mean()
        ss = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0)
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        lr_means = lrs["means"] * (0.01 ** (step / steps))
        for g in opts["means"].param_groups:
            g["lr"] = lr_means
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        strat.step_post_backward(params, opts, state, step, info, lr=lr_means)
        if step % 500 == 0 or step == steps - 1:
            log(f"  step {step}/{steps} loss {loss.item():.4f} n={len(params['means'])} {time.time() - t0:.0f}s")
    stats = dict(train_seconds=time.time() - t0, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                 n_splats=int(len(params["means"])), steps=steps, n_train_frames=len(tr))
    torch.save({k: v.detach().cpu() for k, v in params.items()}, out_dir / "splat.pt")
    export_ply({k: v.detach().cpu() for k, v in params.items()}, out_dir / "splat.ply")
    stats["ply_mb"] = (out_dir / "splat.ply").stat().st_size / 2**20
    return stats


def load_splat(out_dir: Path):
    d = torch.load(Path(out_dir) / "splat.pt")
    return {k: v.to(DEV) for k, v in d.items()}


def export_ply(p, path: Path):
    from plyfile import PlyData, PlyElement

    n = len(p["means"])
    f_dc = p["sh0"].reshape(n, 3).numpy()
    f_rest = p["shN"].transpose(1, 2).reshape(n, -1).numpy()
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([p["means"].numpy(), np.zeros((n, 3), np.float32), f_dc, f_rest,
                           p["opacities"].numpy()[:, None], p["scales"].numpy(),
                           torch.nn.functional.normalize(p["quats"], dim=-1).numpy()], 1).astype(np.float32)
    el = np.empty(n, dtype=[(k, "f4") for k in names])
    for i, k in enumerate(names):
        el[k] = data[:, i]
    PlyData([PlyElement.describe(el, "vertex")]).write(str(path))


# ----------------------------------------------------------------------------- render / eval
@torch.no_grad()
def render_frames(params, frames, out_dir: Path, scale=1.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        W, H = int(f["w"] * scale), int(f["h"] * scale)
        K = np.array(f["K"], np.float32).copy()
        K[0] *= W / f["w"]
        K[1] *= H / f["h"]
        img, _ = _render(params, torch.tensor(f["c2w"], dtype=torch.float32, device=DEV),
                         torch.from_numpy(K).to(DEV), W, H, 3)
        cv2.imwrite(str(out_dir / f["name"]), cv2.cvtColor((img.cpu().numpy() * 255).round().astype(np.uint8),
                                                          cv2.COLOR_RGB2BGR))


@torch.no_grad()
def evaluate(params, frames, save_dir: Path | None = None):
    """PSNR / SSIM / LPIPS of renders vs held-out REAL frames."""
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    net = lpips.LPIPS(net="alex", verbose=False).to(DEV)
    te = [f for f in frames if f["split"] == "test"]
    ps, ss, lp = [], [], []
    for j, f in enumerate(te):
        gt, K = read_image(f)
        H, W = gt.shape[:2]
        pred, _ = _render(params, torch.tensor(f["c2w"], dtype=torch.float32, device=DEV),
                          torch.from_numpy(K).to(DEV), W, H, 3)
        g = torch.from_numpy(gt).to(DEV).float().permute(2, 0, 1)[None] / 255.0
        pr = pred.permute(2, 0, 1)[None]
        ps.append((10 * torch.log10(1.0 / ((pr - g) ** 2).mean())).item())
        ss.append(ssim(pr, g, data_range=1.0).item())
        lp.append(net(pr * 2 - 1, g * 2 - 1).item())
        if save_dir is not None and j % 6 == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            both = np.concatenate([(pr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), gt], 1)
            cv2.imwrite(str(save_dir / f"test_{j:03d}.jpg"), cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
    return dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), lpips=float(np.mean(lp)), n_test=len(te))
