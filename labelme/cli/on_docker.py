#!/usr/bin/env python

from __future__ import print_function

import argparse
import distutils.spawn
import json
import os
import os.path as osp
import platform
import subprocess
import sys


def get_display_host():
    system = platform.system()
    if system == "Linux":
        return os.environ.get("DISPLAY", ":0"), ""
    if system in ["Darwin", "Windows"]:
        host = "host.docker.internal"
        return "{}:0".format(host), host
    raise RuntimeError("Unsupported platform: {}".format(system))


def labelme_on_docker(in_file, out_file):
    display, xhost_target = get_display_host()
    if xhost_target and distutils.spawn.find_executable("xhost"):
        subprocess.check_output(["xhost", "+", xhost_target])
    elif not xhost_target and distutils.spawn.find_executable("xhost"):
        subprocess.check_output(["xhost", "+local:docker"])

    if out_file:
        out_file = osp.abspath(out_file)
        if osp.exists(out_file):
            raise RuntimeError("File exists: %s" % out_file)
        else:
            open(osp.abspath(out_file), "w").close()

    in_file_a = osp.abspath(in_file)
    in_file_b = osp.join("/home/developer", osp.basename(in_file))
    cmd = [
        "docker",
        "run",
        "-it",
        "--rm",
        "-e",
        "DISPLAY={}".format(display),
        "-e",
        "QT_X11_NO_MITSHM=1",
        "-v",
        "{}:{}".format(in_file_a, in_file_b),
        "-w",
        "/home/developer",
    ]
    if osp.isdir("/tmp/.X11-unix"):
        cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix"])
    if out_file:
        out_file_a = osp.abspath(out_file)
        out_file_b = osp.join("/home/developer", osp.basename(out_file))
        cmd.extend(["-v", "{}:{}".format(out_file_a, out_file_b)])
    cmd.extend(["wkentaro/labelme", "labelme", in_file_b])
    if out_file:
        cmd.extend(["-O", out_file_b])
    subprocess.call(cmd)

    if out_file:
        try:
            json.load(open(out_file))
            return out_file
        except Exception:
            if open(out_file).read() == "":
                os.remove(out_file)
            raise RuntimeError("Annotation is cancelled.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input file or directory.")
    parser.add_argument("-O", "--output")
    args = parser.parse_args()

    if not distutils.spawn.find_executable("docker"):
        print("Please install docker", file=sys.stderr)
        sys.exit(1)

    try:
        out_file = labelme_on_docker(args.in_file, args.output)
        if out_file:
            print("Saved to: %s" % out_file)
    except RuntimeError as e:
        sys.stderr.write(e.__str__() + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
