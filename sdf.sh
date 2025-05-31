#!/bin/bash
source activate metricgrids
{
  python main.py --config configs/sdf.py --path ./data/sdf/Armadillo_nrml.obj --alias Armadillo --ds_device cpu
  python main.py --config configs/sdf.py --path ./data/sdf/Bunny_nrml.obj --alias Bunny --ds_device cpu
  python main.py --config configs/sdf.py --path ./data/sdf/Dragon_nrml.obj --alias Dragon --ds_device cpu
  python main.py --config configs/sdf.py --path ./data/sdf/Buddha_nrml.obj --alias Buddha --ds_device cpu
  python main.py --config configs/sdf.py --path ./data/sdf/Lucy_nrml.obj --alias Lucy --ds_device cpu
  python main.py --config configs/sdf.py --path ./data/sdf/XYZDragon_nrml.obj --alias XYZDragon --ds_device cpu
  python main.py --config configs/sdf.py --path ./data/sdf/Statuette_nrml.obj --alias Statuette --ds_device cpu

}