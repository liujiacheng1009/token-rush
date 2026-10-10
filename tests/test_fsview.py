import os
import subprocess

import pytest

from tokenrush.fsview import FsError, list_dir, read_text, run_git, write_text


def test_list_write_and_refuse_outside(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENRUSH_FS_ROOTS", str(tmp_path))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.txt").write_text("hi", encoding="utf-8")
    (proj / "sub").mkdir()
    listed = list_dir(str(proj))
    names = [e["name"] for e in listed["entries"]]
    assert names == ["sub", "a.txt"]
    assert listed["parent"] == str(tmp_path)

    written = write_text(str(proj / "b.txt"), "there", str(proj))
    assert read_text(written) == "there"
    with pytest.raises(FsError):
        write_text(str(tmp_path / "nope.txt"), "x", str(proj))
    with pytest.raises(FsError):
        list_dir("/etc")
    assert list_dir("")["path"] == str(tmp_path)


def test_git_stays_in_the_opened_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENRUSH_FS_ROOTS", str(tmp_path))
    proj = tmp_path / "proj"
    proj.mkdir()
    subprocess_env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE"):
        subprocess_env.pop(name, None)
    subprocess.run(["git", "init"], cwd=proj, check=True, capture_output=True, env=subprocess_env)
    (proj / "a.txt").write_text("hi", encoding="utf-8")
    assert "a.txt" in run_git(str(proj), ["status", "--short"])
    with pytest.raises(FsError):
        run_git(str(proj), ["-C", "/etc", "status"])
    with pytest.raises(FsError):
        run_git(str(proj), ["config", "--global", "user.name", "x"])
    with pytest.raises(FsError):
        run_git(str(proj), ["clone", "https://example.com/nope.git"])
