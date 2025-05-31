#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
source activate metricgrids
{
  python nerfacc/examples/train_ngp_nerf_occ.py --scene chair --data_root data/nerf_synthetic
  python nerfacc/examples/train_ngp_nerf_occ.py --scene drums --data_root data/nerf_synthetic
  python nerfacc/examples/train_ngp_nerf_occ.py --scene ficus --data_root data/nerf_synthetic
  python nerfacc/examples/train_ngp_nerf_occ.py --scene hotdog --data_root data/nerf_synthetic
  python nerfacc/examples/train_ngp_nerf_occ.py --scene lego --data_root data/nerf_synthetic
  python nerfacc/examples/train_ngp_nerf_occ.py --scene materials --data_root data/nerf_synthetic
  python nerfacc/examples/train_ngp_nerf_occ.py --scene mic --data_root data/nerf_synthetic
  python nerfacc/examples/train_ngp_nerf_occ.py --scene ship --data_root data/nerf_synthetic
}
