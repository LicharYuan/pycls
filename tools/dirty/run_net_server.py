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
from NasPred.utils import save_json, load_json, load_pkl, save_pkl

import numpy as np
import copy
import math
# from NasPred.archEncoder.custom_enc import CustomEncoder

def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")

    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    parser.add_argument("--pkl", help="sample net pkl", required=True, type=str)
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
    config.load_cfg(args.cfg)
    cfg.merge_from_list(args.opts)

    # update new net
    ori_pkl = load_pkl(args.pkl)
    post_process = {} # get post net 

    for key in ori_pkl.keys():
        args.net = key
        new_net, net_list = parse_net(args.net)
        new_net_str = "_".join(str(ele) for ele in net_list)
        post_process[new_net_str] = ori_pkl[key]

        cfg["ANYNET"].update(new_net)
        print(cfg)
            
        config.assert_cfg()
        cfg.freeze()

        builders.get_model()()

        complexity = {"complexity" : net.complexity(builders.get_model())}
        print(complexity)

        post_process[new_net_str].update(complexity)

    save_pkl("./post_bench.pkl", post_process)



if __name__ == "__main__":
    main()
    # todo: rm dirty works after experiments
