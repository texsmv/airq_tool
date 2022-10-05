from app_server import initServer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', dest='path', type=str, help='Insert the full path to the mts file')
args = parser.parse_args()


initServer(args.path, host = "127.0.0.1", port=5000)
