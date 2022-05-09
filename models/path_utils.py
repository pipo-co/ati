import os


def get_extension(path: str) -> str:
    return os.path.splitext(path)[1]

def strip_extension(path: str) -> str:
    return os.path.splitext(path)[0]

def append_to_filename(filename: str, s: str) -> str:
    split = os.path.splitext(filename)
    return split[0] + s + split[1]

def lower_extension(path: str) -> str:
    split_name = os.path.splitext(os.path.basename(path))
    return split_name[0] + split_name[1].lower()

