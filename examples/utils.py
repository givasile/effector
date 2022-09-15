import sys
import os

def add_parent_path():
    cwd = os.getcwd()
    if os.path.split(os.getcwd())[-1] == "mdale":
        path_to_examples = os.path.join(cwd, "examples/")
        sys.path.append(cwd)
    elif os.path.split(os.getcwd())[-1] == "examples":
        path_to_examples = cwd
        sys.path.append(os.path.split(cwd)[0])
    else:
        raise OSError

    return path_to_examples
