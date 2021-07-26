from NasPred.query.custom_query import CustomServer
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("host", type=str)
    parser.add_argument("--port", type=int, help="random seed", default=8008)
    parser.add_argument("--sc", type=str, help="shell_scipt", default="run_net.sh")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    host = args.host
    port = args.port
    shell_script = args.sc
    
    shell_script_name  = shell_script.split("/")[-1].split(".")[0]
    query_file = f"./{host}_{port}_{shell_script_name}.json"
    
    server = CustomServer(host, port, query_file)
    server.bind_sh(shell_script, query_file)
    server.run()

