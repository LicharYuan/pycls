from NasPred.query.custom_query import CustomServer
import sys

if __name__ == "__main__":
    # host = "10.10.92.25"
    # host = "127.0.0.1"
    host = sys.argv[1]
    port = 8008
    # port = int(sys.argv[2])
    query_file = f"./{host}_{port}.json"
    shell_script = "run_net.sh"

    server = CustomServer(host, port, query_file)
    server.bind_sh(shell_script, query_file)
    server.run()

