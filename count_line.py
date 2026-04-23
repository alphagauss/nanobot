from pathlib import Path

# 可统计的代码文件后缀
CODE_EXTS = {".py", ".sh", ".js", ".ts"}

# 需要排除的目录
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "tests",

    # "channels",
    # "cli",
    # "command",
    # "providers",
    # "skills"
}

# 各语言的单行注释前缀
COMMENT_PREFIX = {
    ".py": "#",
    ".sh": "#",
    ".js": "//",
    ".ts": "//",
}

def count_code_lines(root="."):
    stats = {
        "files": 0,
        "physical": 0,
        "blank": 0,
        "comment": 0,
        "code": 0,  # 有效代码行
    }

    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue

        if path.suffix not in CODE_EXTS:
            continue

        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue

        comment_prefix = COMMENT_PREFIX.get(path.suffix)

        try:
            with path.open(encoding="utf-8") as f:
                stats["files"] += 1

                for line in f:
                    stats["physical"] += 1
                    stripped = line.strip()

                    if not stripped:
                        stats["blank"] += 1
                    elif comment_prefix and stripped.startswith(comment_prefix):
                        stats["comment"] += 1
                    else:
                        stats["code"] += 1

        except Exception:
            # 编码 / 权限问题直接跳过
            pass

    return stats


if __name__ == "__main__":
    result = count_code_lines()

    print(f"Files           : {result['files']}")
    print(f"Physical lines  : {result['physical']}")
    print(f"Blank lines     : {result['blank']}")
    print(f"Comment lines   : {result['comment']}")
    print(f"Code lines      : {result['code']}")
