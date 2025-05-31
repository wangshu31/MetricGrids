#!/bin/bash
source activate metricgrids
{
  for i in {1..24};
  do
    python main.py --config configs/img.py --path './data/img/Kodak/'$i'.png' \
     --log2_hashmap_size_ref 14 \
     --alias 'Kodak'$i'-paper'
  done
}