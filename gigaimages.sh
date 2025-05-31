#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
source activate metricgrids

{
  python main.py --config configs/img_giga.py --path './data/img/Station.jpg' \
   --log2_hashmap_size_ref 24  --alias 'Station'
  python main.py --config configs/img_giga.py --path './data/img/TheViewatGeestbrug.jpg' \
   --log2_hashmap_size_ref 24  --alias 'TheViewatGeestbrug'
  python main.py --config configs/img_giga.py --path './data/img/tokyo_6k.jpg' \
   --log2_hashmap_size_ref 24  --alias 'tokyo_6k'
  python main.py --config configs/img_giga.py --path './data/img/albert.jpg' \
   --log2_hashmap_size_ref 24  --alias 'albert'
  python main.py --config configs/img_giga.py --path './data/img/pluto.png' \
   --log2_hashmap_size_ref 24  --alias 'pluto'
  python main.py --config configs/img_giga.py --path './data/img/Pearl.jpg' \
   --log2_hashmap_size_ref 24  --alias 'Pearl'
  python main.py --config configs/img_giga.py --path './data/img/tokyo2.jpg'  \
   --log2_hashmap_size_ref 24  --alias 'tokyo2'

} 2>&1 | stdbuf -oL grep -v '%' | tee log/Giga.log
