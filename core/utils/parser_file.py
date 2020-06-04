import os
import sys
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train pipeline')
    parser.add_argument('--num_worker', help='number of worker', type=int)
    parser.add_argument('--hostfile', help='hostfile path', type=str, default="/root/.dist/hostfiles/hostfile")

    args = parser.parse_args()
    return args.num_worker, args.hostfile

def read_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def write_file(hostnames, filepath):
    with open(filepath, "w") as f:
        for name in hostnames:
            f.write(name)
            f.write("\n")

def get_hostname_list(host_dir):
    hostfiles = glob.glob(os.path.join(host_dir, "*.host"))
    hostnames = []
    for i, hostfile in enumerate(hostfiles):
        hostnames = hostnames + read_file(hostfile)
    return hostnames

def generate_hostfile(src_dir, hostfile_path, num_worker):
    hostnames = get_hostname_list(src_dir)

    dst_dir = os.path.dirname(hostfile_path)

    os.makedirs(dst_dir, exist_ok=True)
    assert len(hostnames) == num_worker, "parsered hostname({}) count: {}, required {}".format(hostnames, len(hostnames), num_worker)
    write_file(hostnames, hostfile_path)
    return hostnames

if __name__ == "__main__":
    num_worker, hostfile_path = parse_args()

    host_dir = "/etc/volcano"
    hostnames = generate_hostfile(host_dir, hostfile_path, num_worker)

    print("*********** genereate hotsfile is done **************")
    print("hostnames:{} save to {}".format(hostnames, hostfile_path))

