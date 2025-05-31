"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS

from radiance_fields.ngp import NGPRadianceField

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
import torchvision.transforms as T
sys.path.append('./')
from ssim import ssim_func

from examples.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator


def run(args):
    device = "cuda:0"
    set_random_seed(42)

    if args.scene in MIPNERF360_UNBOUNDED_SCENES:
        from datasets.nerf_360_v2 import SubjectLoader

        # training parameters
        max_steps = 20000
        init_batch_size = 1024
        target_sample_batch_size = 1 << 18
        weight_decay = 0.0
        # scene parameters
        aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
        near_plane = 0.2
        far_plane = 1.0e10
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
        grid_resolution = 128
        grid_nlvl = 4
        # render parameters
        render_step_size = 1e-3
        alpha_thre = 1e-2
        cone_angle = 0.004

    else:
        from datasets.nerf_synthetic import SubjectLoader

        # training parameters
        max_steps = 30000
        init_batch_size = 1024
        target_sample_batch_size = int((1 << 18))
        # weight_decay = 0.0
        weight_decay = (
            1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
        )
        # scene parameters
        aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
        near_plane = 0.0
        far_plane = 1.0e10
        # dataset parameters
        train_dataset_kwargs = {}
        test_dataset_kwargs = {}
        # model parameters
        grid_resolution = 128 # 128
        grid_nlvl = 1
        # render parameters
        render_step_size = 5e-3
        alpha_thre = 0.0
        cone_angle = 0.0

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

    if args.vdb:
        from fvdb import sparse_grid_from_dense

        from nerfacc.estimators.vdb import VDBEstimator

        assert grid_nlvl == 1, "VDBEstimator only supports grid_nlvl=1"
        voxel_sizes = (aabb[3:] - aabb[:3]) / grid_resolution
        origins = aabb[:3] + voxel_sizes / 2
        grid = sparse_grid_from_dense(
            1,
            (grid_resolution, grid_resolution, grid_resolution),
            voxel_sizes=voxel_sizes,
            origins=origins,
        )
        estimator = VDBEstimator(grid).to(device)
        estimator.aabbs = [aabb]
    else:
        estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
        ).to(device)
    n_params = sum([p.numel() for p in estimator.parameters()])
    print(f"No. of estimator parameters: {n_params}")

    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)
    n_params = sum([p.numel() for p in radiance_field.parameters()])
    print(f"No. of radiance field parameters: {n_params}")
    grad_vars = radiance_field.get_optparam_groups(lr_init_grid=2e-2, lr_init_network=1e-3)
    optimizer = torch.optim.Adam(grad_vars,eps=1e-15,weight_decay=weight_decay,)
    # optimizer = torch.optim.Adam(
    #     radiance_field.parameters(),
    #     lr=(
    #         5e-4 if args.scene in ["ship"] else 5e-3
    #     ), # 1e-2
    #     eps=1e-15,
    #     weight_decay=weight_decay,
    # )
    print(f'Optimizer INFO:')
    print(optimizer)
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
    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    # training
    tic = time.time()
    for step in range(max_steps + 1):
        radiance_field.train()
        estimator.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,  # 1e-2
        )

        # render
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)

        # compute loss
        # loss = F.smooth_l1_loss(rgb, pixels)
        loss = F.mse_loss(rgb, pixels)

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if step % 2000 == 0:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | psnr={psnr:.2f} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                f"max_depth={depth.max():.3f} | "
            )

        if step > 0 and step % max_steps == 0:
            # evaluation
            radiance_field.eval()
            estimator.eval()

            psnrs = []
            ssims = []
            lpips = []
            save_dir = f"log/blender-30k/{args.scene}"
            os.makedirs(save_dir, exist_ok=True)

            with torch.no_grad():
                for i in tqdm.tqdm(range(len(test_dataset))):
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]


                    rgb, acc, depth, _ = render_image_with_occgrid(
                        radiance_field,
                        estimator,
                        rays,
                        # rendering options
                        near_plane=near_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                    )
                    mse = F.mse_loss(rgb, pixels) # shape[H,W,3]
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
                # 'lpips_avg': np.mean(lpips)
            }, os.path.join(save_dir, 'model_checkpoint.pth'))
            print(f"INFO:model and testset save to  {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        # default=str(pathlib.Path.cwd() / "data/360_v2"),
        default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
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
        "--vdb",
        action="store_true",
        help="use VDBEstimator instead of OccGridEstimator",
    )
    args = parser.parse_args()

    run(args)
