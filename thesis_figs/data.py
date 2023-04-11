import contextlib
import lzma
import os
import pathlib
import pickle
import re
import shutil
import tempfile

import paths


@contextlib.contextmanager
def open_writable(path, binary=False):
    path = pathlib.Path(path)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True)

    mode = 'wb' if binary else 'w'
    with tempfile.NamedTemporaryFile(mode, dir=str(parent), delete=False) as tf:

        try:
            tempname = pathlib.Path(tf.name)
            yield tf
            tf.close()
            tempname.chmod(0o644)
            tempname.replace(path)
        except:
            tf.close()
            os.unlink(tf.name)
            raise


def has(name):
    pickle_file = paths.DATA_ROOT / f'{name}.pickle'
    xz_file = paths.DATA_ROOT / f'{name}.pickle.xz'
    return pickle_file.exists() or xz_file.exists()


def find(base, pattern='.*'):
    regex = re.compile(rf'({pattern})\.pickle(\.xz)?')
    folder = paths.DATA_ROOT / base
    return filter(
        lambda p: regex.fullmatch(p.relative_to(folder)),
        folder.rglob('*'),
    )


def save(name, data, compress=True):
    relpath = f'{name}.pickle'
    output_file = paths.DATA_ROOT / (relpath + ('.xz' if compress else ''))

    with open_writable(output_file, binary=True) as fp:
        if compress:
            with lzma.open(fp, 'wb') as lzma_file:
                pickle.dump(data, lzma_file, protocol=4)
        else:
            pickle.dump(data, fp, protocol=4)

    if compress:
        with contextlib.suppress(FileNotFoundError):
            (paths.DATA_CACHE_ROOT / relpath).unlink()


def load(name):
    relpath = str(name) + '.pickle'
    pickle_file = paths.DATA_ROOT / relpath

    if pickle_file.exists():
        fp = pickle_file.open('rb')
    else:
        cache = paths.DATA_CACHE_ROOT / relpath
        xz_file = str(pickle_file) + '.xz'

        if cache.exists() and cache.stat().st_mtime < os.stat(xz_file).st_mtime:
            cache.unlink()

        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            with lzma.open(xz_file) as fsrc:
                with open_writable(cache, binary=True) as fdst:
                    shutil.copyfileobj(fsrc, fdst)

        fp = cache.open('rb')

    with fp:
        return pickle.load(fp)
