"""
Register the NoRGa learner in utils/factory.py (idempotent; writes factory.py.bak first).

    python3 scripts/norga/register_norga.py            # patches utils/factory.py
    python3 scripts/norga/register_norga.py --dry-run  # prints the patched function head

It inserts, at the top of get_model(...), a branch for "norga", "promptcl" and
"hideprompt" that returns models.norga.Learner. Placing the branch first means it works
whatever style the rest of get_model uses (if/elif chain or a dict lookup).
"""
import argparse
import pathlib
import re
import shutil
import sys

MARKER = "models.norga"


def patch(src):
    if MARKER in src:
        return None, "already registered"
    m = re.search(r"^def get_model\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*:[^\n]*\n", src, re.M)
    if not m:
        return None, "could not find `def get_model(name, args):`"
    name_arg, args_arg = m.group(1), m.group(2)
    rest = src[m.end():]
    indent_m = re.match(r"(?:[ \t]*\n)*([ \t]+)\S", rest)
    ind = indent_m.group(1) if indent_m else "    "
    branch = (
        f"{ind}# NoRGa / HiDe-Prompt prompt-based baseline (see NORGA.md)\n"
        f"{ind}if str({name_arg}).lower() in (\"norga\", \"promptcl\", \"hideprompt\"):\n"
        f"{ind}    from models.norga import Learner as _NoRGaLearner\n"
        f"{ind}    return _NoRGaLearner({args_arg})\n"
    )
    return src[:m.end()] + branch + rest, "patched"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("factory", nargs="?", default="utils/factory.py")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()
    path = pathlib.Path(opts.factory)
    if not path.exists():
        sys.exit(f"{path} not found; run from the repository root")
    new_src, status = patch(path.read_text())
    if new_src is None:
        print(f"{path}: {status}")
        sys.exit(0 if status == "already registered" else 1)
    if opts.dry_run:
        start = new_src.index("def get_model")
        print(new_src[start:start + 600])
        return
    shutil.copy(path, str(path) + ".bak")
    path.write_text(new_src)
    print(f"{path}: {status} (backup at {path}.bak)")


if __name__ == "__main__":
    main()
