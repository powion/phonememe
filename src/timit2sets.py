"Split TIMIT corpus into *.npy files for training, validation and test sets."


from __future__ import print_function, unicode_literals

import sys
import bz2
import time
import pickle
import argparse
import subprocess
from bisect import bisect_left
from os import walk, path, listdir
from itertools import groupby, chain
from collections import deque
from multiprocessing import Pool

import numpy as np


def file_extension(fn):
    return path.splitext(fn)[0]


class TIMIT(object):
    "TIMIT corpus API"

    _dataset_names = ('test', 'train')

    def __init__(self, base):
        self.base = base
        self.subpath = lambda *p: path.join(base, *p)

    def __iter__(self):
        return self.all()

    def all(self):
        return chain.from_iterable(map(self.dataset, self._dataset_names))

    def dataset(self, name):
        for region in listdir(self.subpath(name)):
            for speaker in listdir(self.subpath(name, region)):
                p = self.subpath(name, region, speaker)
                for key, g in groupby(listdir(p), file_extension):
                    yield Sample(self.base, name, region, speaker, key)

    def build_phoneme_map(self):
        phoneme_map = {}
        for sample in self:
            for start, stop, phoneme in sample.phonemes():
                phoneme_map.setdefault(phoneme, len(phoneme_map))
        return phoneme_map


class Sample(object):
    def __init__(self, base, dataset, region, speaker, key):
        self.base = base
        self.dataset = dataset
        self.region = region
        self.speaker = speaker
        self.key = key
        self.path = path.join(base, dataset, region, speaker, key)
        self.part = lambda ext: '{}.{}'.format(self.path, ext)

    def phonemes(self):
        with open(self.part('phn')) as f:
            for line in f:
                start, stop, phoneme = line.split()
                yield int(start), int(stop), phoneme

    def labelled_frames(self, length, step=3, frame_size=160):
        phonemes = deque(self.phonemes())
        first_phoneme_start, _, _ = phonemes[0]
        _, last_phoneme_stop, _   = phonemes[-1]
        first_frame = first_phoneme_start//frame_size
        last_frame  = last_phoneme_stop//frame_size
        start, stop, phoneme = None, None, None
        for frame in range(first_frame, last_frame, step):
            pos = frame*frame_size
            while phonemes and stop < pos:
                start, stop, phoneme = phonemes.popleft()
            if start <= pos:
                yield (max(0, frame - length), frame, phoneme)


def wav2fbank(fn):
    out = subprocess.check_output('HList -C htk.cfg -r'.split() + [fn])
    return np.array([map(float, line.split()) for line in out.splitlines()])


def iter_pairs(ds, maxlen=20):
    for sample in ds:
        feats = wav2fbank(sample.part('wav'))
        for start, stop, phoneme in sample.labelled_frames(maxlen):
            pad = np.zeros((maxlen + start - stop, feats.shape[1]))
            v = np.concatenate((pad, feats[start:stop, :]))
            yield (v, phoneme)


def flowrated(it, clk=time.time, interval=10.0,
              fmt='{rate:.3g} o/s, {n} total'.format):
    t0, np = clk(), 0
    for n, v in enumerate(it):
        t1 = clk()
        yield v
        if t1 - t0 > interval:
            rate = (n - np) / (t1 - t0)
            print(fmt(rate=rate, n=n), file=sys.stderr)
            t0, np = clk(), n


def create_dataset(corpus, ds, phoneme_map_fn='phoneme_map.npy'):
    if not path.exists(phoneme_map_fn):
        phoneme_map = corpus.build_phoneme_map()
        print('writing phoneme map:', phoneme_map_fn)
        np.save(phoneme_map_fn, phoneme_map)
    else:
        print('reusing phoneme map:', phoneme_map_fn)
        phoneme_map = np.load(phoneme_map_fn).tolist()
    X, y = [], []
    for feats, phoneme in iter_pairs(flowrated(corpus.dataset(ds))):
        X.append(feats)
        y.append(phoneme_map[phoneme])
    X, y = np.array(X), np.array(y)
    np.memmap('X_{}.mat'.format(ds), dtype='float32', mode='w+', shape=X.shape)[:] = X[:]
    np.memmap('y_{}.mat'.format(ds), dtype='float32', mode='w+', shape=y.shape)[:] = y[:]
    print(X.shape); print(y.shape)


def main(args=sys.argv[1:]):
    corpus, ds = TIMIT(args[0]), args[1]
    create_dataset(corpus, ds)


if __name__ == "__main__":
    main()
