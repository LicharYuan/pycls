#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute various operations (train, test, time, etc.) on a classification model."""

import argparse
import sys

import pycls.core.builders as builders
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.net as net
import pycls.core.trainer as trainer
import pycls.models.scaler as scaler
from pycls.core.config import cfg

from NasPred.query.custom_query import CustomServer
from NasPred.utils import save_json, load_json

import numpy as np
import copy
import math
import time
# from NasPred.archEncoder.custom_enc import CustomEncoder

def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s, choices = "Run mode", ["info", "train", "test", "time", "scale"]
    parser.add_argument("--mode", help=help_s, choices=choices, required=True, type=str)
    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    parser.add_argument("--net", help="sample net", required=True, type=str)
    parser.add_argument("--query_file", help="query file for remote", required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def get_all_divs(n):
    res = []
    i = 1
    while i <= n:
        if (n%i == 0):
            res.append(i)
        i += 1

    return res

def match_divs(alist, nums_list, metric="abs"):
    res = []
    for ele, num in zip(alist, nums_list):
        
        all_divs = get_all_divs(ele) # sorted

        if num in all_divs:
            res.append(num)
        else:
            all_divs = np.array(all_divs)
            if metric == "abs":
                dis = np.abs(all_divs - num)
                nearst_idx = dis.argsort()[0]

            elif metric in ["ceil", "floor"]:
                dis = all_divs > num
                dis = dis.tolist()
                nearst_idx = dis.index(1) if metric=="ceil" else dis.index(1) - 1
            
            res.append(all_divs[nearst_idx])

    return res



def parse_net(net):
    net_list = [eval(ele) for ele in net.strip().split("_")]
    depths = net_list[:4]
    widths = net_list[4:8]
    bot_mults = net_list[8:12]
    bot_mults = (1 / np.array(bot_mults)).tolist()

    groups = net_list[12:16]

    w_out = np.array(widths) * np.array(bot_mults) 
    # group should be divisible of w_out
    # here need post-porcess

    groups = match_divs(w_out, groups)
    net_list[12:16] = groups
        
    groups_ws = np.array(widths) * np.array(bot_mults) // np.array(groups)
        
    groups_ws = [int(ele) for ele in groups_ws.tolist()]

    new_net = {
        "DEPTHS": depths,
        "WIDTHS": widths,
        "BOT_MULS": bot_mults,
        "GROUP_WS": groups_ws,
    }

    return new_net, net_list


def main():
    """Execute operation (train, test, time, etc.)."""
    args = parse_args()
    mode = args.mode
    config.load_cfg(args.cfg)
    cfg.merge_from_list(args.opts)

    # update new net
    new_net, net_list = parse_net(args.net)

    cfg["ANYNET"].update(new_net)
    cfg["NET_POST"] = "_".join(str(ele) for ele in net_list)
    cfg["NET_ORI"] = args.net.strip()
    cfg["QUERY_FILE"] = args.query_file.strip()
    query_data = load_json(args.query_file)
    
    # TODO: mv post-process in sampler! 
    # 有一种采样case, 两个网络对应同一个后处理的结构 会重复处理, 但是概率比较低
    # 处理方式, 在query_file把后处理的的网络加进去, 作为flag. 当前未添加.
    
    config.assert_cfg()
    cfg.freeze()
    if mode == "info":
        print(builders.get_model()())
        comp = net.complexity(builders.get_model())
        stand_comp = {}
        for key, value in comp.items():
            if key == "flops":
                stand_comp[key] = value / 1e9
            elif key == "params":
                stand_comp[key] = value / 1e6
            elif key == "acts":
                stand_comp[key] = value / 1e6

        net_info = {"complexity": stand_comp}
        print(net_info)
        query_data[args.net.strip()] = net_info 
        save_json(args.query_file, query_data)
        time.sleep(1)

    elif mode == "train":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)
    elif mode == "test":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.test_model)
    elif mode == "time":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.time_model)
    elif mode == "scale":
        cfg.defrost()
        cx_orig = net.complexity(builders.get_model())
        scaler.scale_model()
        cx_scaled = net.complexity(builders.get_model())
        cfg_file = config.dump_cfg()
        print("Scaled config dumped to:", cfg_file)
        print("Original model complexity:", cx_orig)
        print("Scaled model complexity:", cx_scaled)



if __name__ == "__main__":
    main()
