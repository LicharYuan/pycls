#!/bin/bash
QUERY_FILE=$1
NET=$2
FLOPS=$3
CONFIG=$4
echo  $NET
echo  $QUERY_FILE
echo  $CONFIG
echo "RUNNING TRAIN"
./tools/run_net_server.py --mode train \
                          --query_file $QUERY_FILE \
                          --cfg ${CONFIG} \
                          --net $NET \
                          OUT_DIR ./ts/Search/calibirate/${NET}_${FLOPS}

# echo "RUNNING TEST"
# ./tools/run_net_server.py --mode test --query_file ./_tmp.json --cfg configs/dds_baselines/resnet/any_dds_8gpu_base.yaml --net $NET OUT_DIR ./ts/AnyNet_$NET TEST.WEIGHTS ./ts/AnyNet_$NET/checkpoints/model_epoch_0001.pyth 
