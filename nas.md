
## flops query 

`python run_server.py 10.10.81.88 --port 8001 --sc run_net_info.sh --qf ./query/info.json  `

## retrain run

**< 3.2 GF**

`sh run_search_net.sh ./num30.json 4_1_8_11_128_240_472_968_2_2_2_2_7_11_31_9  1.64GF  ./configs/dds_baselines/resnet/any_dds_8gpu_full_base.yaml`

**>=3.2GF**

``sh run_search_net.sh ./num30.json $NET$  $NET_FLOPS$  ./configs/dds_baselines/resnet/any_dds_8gpu_full_large_base.yaml`
`













