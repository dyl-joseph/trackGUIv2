#!/usr/bin/env python

from __future__ import print_function

import argparse
import json
import os
import os.path as osp
import platform
import shutil
import subprocess
import sys
import tempfile


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
    xhost_rule = xhost_target if xhost_target else "local:docker"
    xhost_enabled = bool(shutil.which("xhost"))

    temporary_output = None
    if out_file:
        out_file = osp.abspath(out_file)
        if osp.exists(out_file):
            raise RuntimeError("File exists: %s" % out_file)
        fd, temporary_output = tempfile.mkstemp(
            dir=osp.dirname(out_file) or None,
            prefix=".{}-".format(osp.basename(out_file)),
            suffix=".tmp",
        )
        os.close(fd)

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
        out_file_a = temporary_output
        out_file_b = osp.join("/home/developer", osp.basename(out_file))
        cmd.extend(["-v", "{}:{}".format(out_file_a, out_file_b)])
    cmd.extend(["wkentaro/labelme", "labelme", in_file_b])
    if out_file:
        cmd.extend(["-O", out_file_b])
    try:
        if xhost_enabled:
            subprocess.run(["xhost", "+{}".format(xhost_rule)], check=True)
        subprocess.run(cmd, check=True)
        if out_file:
            try:
                with open(temporary_output, encoding="utf-8") as handle:
                    json.load(handle)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Annotation is cancelled or invalid.") from exc
            os.replace(temporary_output, out_file)
            temporary_output = None
            return out_file
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Docker annotation failed: {}".format(exc)) from exc
    finally:
        if xhost_enabled:
            subprocess.run(
                ["xhost", "-{}".format(xhost_rule)],
                check=False,
            )
        if temporary_output and osp.exists(temporary_output):
            os.unlink(temporary_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input file or directory.")
    parser.add_argument("-O", "--output")
    args = parser.parse_args()

    if not shutil.which("docker"):
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
