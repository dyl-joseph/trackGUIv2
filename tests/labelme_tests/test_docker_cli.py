import json
import subprocess
from pathlib import Path

import pytest

from labelme.cli import on_docker


def _mounted_output_path(command, destination_name):
    for index, argument in enumerate(command):
        if argument == "-v":
            source, destination = command[index + 1].split(":", 1)
            if destination.endswith(destination_name):
                return Path(source)
    raise AssertionError("output mount was not present")


def test_docker_helper_revokes_xhost_and_atomically_installs_output(
    monkeypatch, tmp_path
):
    input_file = tmp_path / "frame.jpg"
    input_file.write_bytes(b"image")
    output_file = tmp_path / "frame.json"
    commands = []

    monkeypatch.setattr(on_docker, "get_display_host", lambda: (":0", ""))
    monkeypatch.setattr(on_docker.shutil, "which", lambda _: "xhost")

    def fake_run(command, check):
        commands.append(command)
        if command[0] == "docker":
            temporary = _mounted_output_path(command, output_file.name)
            temporary.write_text(json.dumps({"shapes": []}), encoding="utf-8")

    monkeypatch.setattr(on_docker.subprocess, "run", fake_run)

    assert on_docker.labelme_on_docker(str(input_file), str(output_file)) == str(
        output_file
    )
    assert json.loads(output_file.read_text(encoding="utf-8")) == {"shapes": []}
    assert commands[0][0:2] == ["xhost", "+local:docker"]
    assert commands[-1][0:2] == ["xhost", "-local:docker"]


def test_docker_helper_removes_temporary_output_after_failure(monkeypatch, tmp_path):
    input_file = tmp_path / "frame.jpg"
    input_file.write_bytes(b"image")
    output_file = tmp_path / "frame.json"
    commands = []

    monkeypatch.setattr(on_docker, "get_display_host", lambda: (":0", ""))
    monkeypatch.setattr(on_docker.shutil, "which", lambda _: "xhost")

    def fake_run(command, check):
        commands.append(command)
        if command[0] == "docker":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(on_docker.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Docker annotation failed"):
        on_docker.labelme_on_docker(str(input_file), str(output_file))

    assert not output_file.exists()
    assert commands[-1][0:2] == ["xhost", "-local:docker"]
    assert list(tmp_path.glob(".frame.json-*.tmp")) == []
