"""
Fineweb-edu dataset for serious pretraining.
Karpathy doesnt have very good results with this dataset. But it might be a setting issue.
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

parser = argparse.ArgumentParser(description="Fineweb-edu dataset preprocessing")
parser.add_argument("-v", "--version", type=str, default="10B", help="Which version of fineweb-edu to use 10B|100B")
parser.add_argument("-s", "--shard-size", type=int, default=10**8, help="Size of each shard in token")
args = parser.parse_args()

# Fineweb-edu has a few possible subsamples available
assert args.version in ["10B", "100B"], "Invalid version of fineweb-edu"
if args.version == "10B":
    local_dir = "finewebedu10B"
    remote_name = "sample-10BT"
elif args.version == "100B":
    local_dir = "finewebedu100B"
    remote_name = "sample-100BT"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fwe = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenize a single document and return a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token to big for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold curretn shard
    all_token_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for token in pool.imap(tokenize, fwe, chunksize=16):
        
        # is there enough space in the current shard for the new token?
        if token_count + len(token) < args.shard_size:
            # simply append the token to the current shard
            all_token_np[token_count:token_count+len(token)] = token
            token_count += len(token)
            # update progress_bar
            if progress_bar is None:
                progress_bar = tqdm(total=len(fwe), unit="tokens", desc=f"shard {shard_index}")
            progress_bar.update(len(token))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"finewebedu_{split}_{shard_index:06d}.bin")
            # split the document into whatever files in this shard; the remainder goes to the enxt one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_token_np[token_count:] = token[:remainder]
            write_datafile(filename, all_token_np)
            shard_index += 1
            progress_bar = None
            # populate the nex5t shard with the leftovers of the current document
            all_token_np[0:len(token)-remainder] = token[remainder:]
            token_count = len(token) - remainder
    
	# write any remaining tokens are the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_token_np[:token_count])