from fastai.text.all import *
from pathlib import Path
import pathlib


from contextlib import contextmanager
import pathlib

@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup



EXPORT_PATH = pathlib.Path("/home/ec2-user/sentiment_analysis/classifier3.pkl")

with set_posix_windows():
    learn_inf = load_learner(EXPORT_PATH)
    print(learn_inf.predict("the movie was fantastic")[0])