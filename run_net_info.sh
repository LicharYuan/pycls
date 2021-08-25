#!/bin/bash
QUERY_FILE=$1
NET=$2
echo  $NET
echo  $QUERY_FILE
echo "RUNNING INFO"
./tools/run_net_server.py --mode info \
                          --query_file $QUERY_FILE \
                          --cfg configs/dds_baselines/resnet/any_dds_8gpu_base_calibrate.yaml \
                          --net $NET \
                          OUT_DIR ./ts/info/AnyNet_$NET 

