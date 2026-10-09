import os

import pytest

from tokenrush.fsview import FsError, list_dir, read_text, write_text


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
