"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import argparse
import itertools
import pathlib
import time
from typing import Callable

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngpprop import NGPDensityField, NGPRadianceField
import os
import sys
from PIL import Image
import torchvision.transforms as T
sys.path.append('./')
from ssim import ssim_func

from examples.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_propnet,
    set_random_seed,
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/360_v2"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 4096
    weight_decay = 0.0
    # scene parameters
    unbounded = True
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2  # TODO: Try 0.02
    far_plane = 1e3
    # dataset parameters
    # train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    # test_dataset_kwargs = {"factor": 4}
    train_dataset_kwargs = {
        "color_bkgd_aug": "random",
        "factor": 4 if args.scene in ["bicycle", "flowers", "garden", "stump", "treehill"] else 2
    }
    test_dataset_kwargs = {
        "factor": 4 if args.scene in ["bicycle", "flowers", "garden", "stump", "treehill"] else 2
    }
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=256,
        ).to(device),
    ]
    # render parameters
    num_samples = 48
    num_samples_per_prop = [256, 96]
    sampling_type = "lindisp"
    opaque_bkgd = True

else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 4096
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    unbounded = False
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 2.0
    far_plane = 6.0
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
    ]
    # render parameters
    num_samples = 64
    num_samples_per_prop = [128]
    sampling_type = "uniform"
    opaque_bkgd = False

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

# setup the radiance field we want to train.
prop_optimizer = torch.optim.Adam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            prop_optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            prop_optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(device)

grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
n_params = sum([p.numel() for p in radiance_field.parameters()])
print(f"No. of radiance field parameters: {n_params}")
optimizer = torch.optim.Adam(
    radiance_field.parameters(),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
proposal_requires_grad_fn = get_proposal_requires_grad_fn()
# proposal_annealing_fn = get_proposal_annealing_fn()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    for p in proposal_networks:
        p.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    proposal_requires_grad = proposal_requires_grad_fn(step)
    # render
    rgb, acc, depth, extras = render_image_with_propnet(
        radiance_field,
        proposal_networks,
        estimator,
        rays,
        # rendering options
        num_samples=num_samples,
        num_samples_per_prop=num_samples_per_prop,
        near_plane=near_plane,
        far_plane=far_plane,
        sampling_type=sampling_type,
        opaque_bkgd=opaque_bkgd,
        render_bkgd=render_bkgd,
        # train options
        proposal_requires_grad=proposal_requires_grad,
    )
    estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )

    # compute loss
    # loss = F.smooth_l1_loss(rgb, pixels)
    loss = F.mse_loss(rgb, pixels)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 5000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and step % max_steps == 0:
        # evaluation
        radiance_field.eval()
        for p in proposal_networks:
            p.eval()
        estimator.eval()

        psnrs = []
        ssims = []
        lpips = []
        save_dir = f"logs/prop-Nov5/{args.scene}"
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                (rgb, acc, depth, _,) = render_image_with_propnet(
                    radiance_field,
                    proposal_networks,
                    estimator,
                    rays,
                    # rendering options
                    num_samples=num_samples,
                    num_samples_per_prop=num_samples_per_prop,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    sampling_type=sampling_type,
                    opaque_bkgd=opaque_bkgd,
                    render_bkgd=render_bkgd,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                ssim = ssim_func(rgb, pixels)
                ssims.append(ssim.item())
                lpips.append(lpips_fn(rgb, pixels).item())


                rgb = rgb.permute(2, 0, 1)
                pixels = pixels.permute(2, 0, 1)
                rgb_image = T.ToPILImage()(rgb.cpu().clamp(0, 1))
                pixels_image = T.ToPILImage()(pixels.cpu().clamp(0, 1))
                error_gray = torch.abs(rgb - pixels)
                error_gray = error_gray.mean(dim=0, keepdim=True)
                error_image = T.ToPILImage()(error_gray.cpu().clamp(0, 1))
                concatenated_image = Image.new('RGB', (rgb_image.width * 3, rgb_image.height))
                concatenated_image.paste(pixels_image, (0, 0))
                concatenated_image.paste(rgb_image, (rgb_image.width, 0))
                concatenated_image.paste(error_image, (rgb_image.width * 2, 0))
                save_path = os.path.join(save_dir, f'test_{i}.png')
                concatenated_image.save(save_path)

        psnr_avg = sum(psnrs) / len(psnrs)
        print(f"INFO:evaluation: PSNR_avg={psnr_avg}")
        ssim_avg = sum(ssims) / len(ssims)
        print(f"INFO:evaluation: SSIM_avg={ssim_avg}")
        lpips_avg = sum(lpips) / len(lpips)
        print(f"INFO:evaluation: lpips_avg={lpips_avg}")
        torch.save({
            'radiance_field': radiance_field.state_dict(),
            'estimator': estimator.state_dict(),
            'step': step,
            'psnr_avg': np.mean(psnrs),
            'lpips_avg': np.mean(lpips)
        }, os.path.join(save_dir, 'model_checkpoint.pth'))
        print(f"INFO:model and testset save to  {save_dir}")
