"""
Adapted from PyTorch's text library.
"""

import array
import os
import zipfile

import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys

def obj_edge_vectors(names, wv_type='glove.6B', wv_dir=None, wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0,1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token.split('/')[0], None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word (hopefully won't be a preposition
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))

    return vectors

URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }


def load_word_vectors(root, wv_type, dim):
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)
    
    pt_file = fname + '.pt'
    if os.path.isfile(pt_file):
        print(f'Loading word vectors from {pt_file}')
        try:
            return torch.load(pt_file)
        except Exception as e:
            print(f"Error loading the model from {pt_file}: {e}")
            raise

    txt_file = fname + '.txt'
    print(f"Looking for text file: {txt_file}")
    lines = []
    if os.path.isfile(txt_file):
        print(f'Loading word vectors from {txt_file}')
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    elif wv_type in URL:
        url = URL[wv_type]
        print(f'Downloading word vectors from {url}')
        filename = os.path.basename(fname)
        os.makedirs(root, exist_ok=True)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print(f'Extracting word vectors into {root}')
                zf.extractall(root)
        if not os.path.isfile(txt_file):
            print(f"File {txt_file} not found after extraction")
            raise RuntimeError('No word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError(f'No URL found for {wv_type}')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if lines:
        for line in tqdm(lines, desc=f"Loading word vectors from {txt_file}"):
            entries = line.strip().split()
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except Exception as e:
                print(f'Non-UTF8 token {repr(word)} ignored. Error: {e}')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    if wv_size is None:
        print("Failed to determine word vector size.")
        raise RuntimeError('Word vector size could not be determined')

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    result = (wv_dict, wv_arr, wv_size)
    torch.save(result, pt_file)
    return result


def reporthook(t):
    """https://github.com/YXblank/scene"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: ĺeftright].
        bsize: int, optional
        Size of each block [default: ĺeftright].
        tsize: int, optional
        Total size . If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
