#!/bin/bash
NET=$1
echo  $NET
echo "RUNNING TRAIN"
./tools/run_net_server.py --mode train --cfg configs/dds_baselines/resnet/any_dds_8gpu_base.yaml --net $NET OUT_DIR ./ts/AnyNet_$NET

echo "RUNNING TEST"
./tools/run_net_server.py --mode train --cfg configs/dds_baselines/resnet/any_dds_8gpu_base.yaml --net $NET OUT_DIR ./ts/AnyNet_$NET TEST.WEIGHTS ./ts/AnyNet_$NET/checkpoints/model_epoch_0010.pyth