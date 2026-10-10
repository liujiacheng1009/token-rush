"""Read and write text files for the code page. Paths stay under the home directory,
or under TOKENRUSH_FS_ROOTS when a test sets it. Git runs only inside the opened
project, and only as the git commands the code page is allowed to use.
"""
import os
import subprocess

_MAX_BYTES = 1_000_000
_LIST_CAP = 500
_READ_CAP = 12_000


class FsError(Exception):
    pass


def roots() -> list[str]:
    raw = os.environ.get("TOKENRUSH_FS_ROOTS")
    if raw:
        return [os.path.realpath(p) for p in raw.split(os.pathsep) if p]
    return [os.path.realpath(os.path.expanduser("~"))]


def resolve(path: str) -> str:
    if not path or not os.path.isabs(path):
        raise FsError("需要绝对路径")
    real = os.path.realpath(path)
    for root in roots():
        if real == root or real.startswith(root + os.sep):
            return real
    raise FsError("这个目录不在允许的范围内")


def inside(path: str, root: str) -> str:
    real = resolve(path)
    base = resolve(root)
    if real == base or real.startswith(base + os.sep):
        return real
    raise FsError("只能改当前打开的目录")


def list_dir(path: str) -> dict:
    real = resolve(path or roots()[0])
    if not os.path.isdir(real):
        raise FsError("不是目录")
    entries = []
    try:
        scanned = list(os.scandir(real))
    except OSError as exc:
        raise FsError("打不开这个目录") from exc
    for ent in scanned:
        if len(entries) >= _LIST_CAP:
            break
        try:
            entries.append({"name": ent.name, "dir": ent.is_dir(follow_symlinks=False)})
        except OSError:
            continue
    entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
    parent = os.path.dirname(real)
    if parent == real:
        parent = None
    else:
        try:
            parent = resolve(parent)
        except FsError:
            parent = None
    return {"path": real, "parent": parent, "entries": entries}


def read_text(path: str, limit: int | None = None) -> str:
    real = resolve(path)
    if os.path.isdir(real):
        raise FsError("这是目录")
    try:
        size = os.path.getsize(real)
    except OSError as exc:
        raise FsError("打不开这个文件") from exc
    if size > _MAX_BYTES:
        raise FsError("文件太大")
    try:
        text = open(real, encoding="utf-8").read()
    except UnicodeDecodeError as exc:
        raise FsError("不是文本文件") from exc
    except OSError as exc:
        raise FsError("打不开这个文件") from exc
    if limit is not None and len(text) > limit:
        return text[:limit] + f"\n…（已截断，后面还有 {len(text) - limit} 字）"
    return text


def write_text(path: str, content: str, root: str) -> str:
    if not isinstance(content, str):
        raise FsError("内容必须是文本")
    real = inside(path, root)
    if os.path.isdir(real):
        raise FsError("这是目录")
    raw = content.encode("utf-8")
    if len(raw) > _MAX_BYTES:
        raise FsError("文件太大")
    parent = os.path.dirname(real)
    if not os.path.isdir(parent):
        raise FsError("上级目录不存在")
    with open(real, "w", encoding="utf-8") as fh:
        fh.write(content)
    return real


_GIT_OK = frozenset({
    "status", "diff", "log", "show", "add", "commit", "push", "pull", "fetch",
    "branch", "checkout", "switch", "restore", "reset", "merge", "rebase",
    "stash", "tag", "rm", "mv", "remote", "rev-parse", "ls-files", "config",
    "init", "clean", "cherry-pick", "revert", "blame", "shortlog",
})
_GIT_BLOCK = ("-C", "--git-dir", "--work-tree", "--exec-path", "-c", "--config-env")
_GIT_CONFIG_BLOCK = ("alias.", "core.sshcommand", "core.hookspath", "core.fsmonitor", "core.pager")


def run_git(root: str, args: list) -> str:
    """Run one git command with the opened project as its working directory."""
    base = resolve(root)
    if not isinstance(args, list) or not args or not all(isinstance(a, str) and a for a in args):
        raise FsError("git 需要一组参数")
    if len(args) > 32 or any(len(a) > 4000 for a in args):
        raise FsError("git 参数太多")
    for arg in args:
        if any(arg == flag or arg.startswith(flag + "=") for flag in _GIT_BLOCK):
            raise FsError("只能在当前打开的目录里执行 git")
        if os.path.isabs(arg):
            inside(arg, base)
    command = next((arg for arg in args if not arg.startswith("-")), "")
    if command not in _GIT_OK:
        raise FsError("不会执行 git " + (command or ""))
    if command == "config" and any(arg in ("--global", "--system") for arg in args):
        raise FsError("只能改当前仓库的 git 配置")
    if command == "config":
        keys = [arg for arg in args if not arg.startswith("-")]
        key = keys[0].lower() if keys else ""
        if key.startswith(_GIT_CONFIG_BLOCK) or key in _GIT_CONFIG_BLOCK:
            raise FsError("不能改这个 git 配置")
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_EXEC_PATH"):
        env.pop(name, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        proc = subprocess.run(
            ["git", *args], cwd=base, env=env, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise FsError("git 超时") from exc
    out = "\n".join(part for part in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if part)
    if len(out) > _READ_CAP:
        out = out[:_READ_CAP] + "\n…（已截断）"
    if proc.returncode != 0:
        raise FsError(out or f"git 退出码 {proc.returncode}")
    return out or "完成"
