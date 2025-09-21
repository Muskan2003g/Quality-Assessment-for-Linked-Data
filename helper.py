# helper.py — robust I/O, logging, saving, and small utilities

import os
import json
import torch
import numpy as np
import logging
import heapq
import random
from typing import Iterable, List, Tuple, Optional


# ----------------------- Small utils -----------------------

class TopKHeap(object):
    """Min-heap that keeps the top-k items by tuple ordering (score first)."""
    def __init__(self, k: int):
        self.k = int(k)
        self.data: List[Tuple] = []

    def push(self, elem: Tuple):
        if self.k <= 0:
            return
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            # python heap is min-heap; compare by first element of tuple (score)
            if elem > self.data[0]:
                heapq.heapreplace(self.data, elem)

    def topk(self) -> List[Tuple]:
        return [x for x in reversed([heapq.heappop(self.data) for _ in range(len(self.data))])]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------------- Config override -----------------------

def override_config(args):
    """
    Override model and data configuration from a saved checkpoint directory.
    Gracefully handle missing optional keys.
    """
    cfg_path = os.path.join(args.init_checkpoint, 'configs.json')
    with open(cfg_path, 'r', encoding='utf-8') as fjson:
        argparse_dict = json.load(fjson)

    # Optional keys
    if 'countries' in argparse_dict:
        args.countries = argparse_dict['countries']

    # Required / common keys (guarded)
    if args.data_path is None and 'data_path' in argparse_dict:
        args.data_path = argparse_dict['data_path']
    if 'model' in argparse_dict:
        args.model = argparse_dict['model']
    if 'double_entity_embedding' in argparse_dict:
        args.double_entity_embedding = argparse_dict['double_entity_embedding']
    if 'double_relation_embedding' in argparse_dict:
        args.double_relation_embedding = argparse_dict['double_relation_embedding']
    if 'hidden_dim' in argparse_dict:
        args.hidden_dim = argparse_dict['hidden_dim']


# ----------------------- Saving -----------------------

def save_model(model, ssvae_model, optimizer, save_variable_list, args, is_best_model=False):
    """
    Save KGE model, optional SSVAE model, optimizer, and misc vars.
    """
    save_path = f"{args.save_path}/best/" if is_best_model else args.save_path
    os.makedirs(save_path, exist_ok=True)

    argparse_dict = vars(args)
    with open(os.path.join(save_path, 'configs.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    # KGE embedding model
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(save_path, 'checkpoint'))

    # Optional SSVAE
    if ssvae_model is not None:
        torch.save({'model_state_dict': ssvae_model.state_dict()},
                   os.path.join(save_path, 'ssvae_checkpoint'))

    # Numpy dumps of embeddings (if present)
    if getattr(model, 'entity_embedding', None) is not None:
        np.save(os.path.join(save_path, 'entity_embedding'),
                model.entity_embedding.detach().cpu().numpy())
    if getattr(model, 'relation_embedding', None) is not None:
        np.save(os.path.join(save_path, 'relation_embedding'),
                model.relation_embedding.detach().cpu().numpy())




# ----------------------- Data reading -----------------------

def _split_triple_line(line: str):
    """
    Split a triple line tolerant of both tab and space separators.
    Accepts formats like 'h<tab>r<tab>t' or 'h r t'.
    """
    line = line.strip()
    if not line:
        return None
    if '\t' in line:
        parts = line.split('\t')
    else:
        parts = line.split()
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def read_triple(file_path: str, entity2id=None, relation2id=None):
    """
    Read triples and map them into ids if dictionaries are provided.
    Otherwise expect numeric IDs in the file.
    Returns: List[Tuple[int,int,int]]
    """
    triples: List[Tuple[int, int, int]] = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as fin:
        for raw in fin:
            parsed = _split_triple_line(raw)
            if parsed is None:
                continue
            h, r, t = parsed
            if entity2id is None or relation2id is None:
                # Expect numbers in the file
                try:
                    triples.append((int(h), int(r), int(t)))
                except ValueError:
                    # Skip malformed numeric lines gracefully
                    continue
            else:
                # Map from labels to ids
                if h in entity2id and r in relation2id and t in entity2id:
                    triples.append((int(entity2id[h]), int(relation2id[r]), int(entity2id[t])))
                else:
                    # Skip unknown tokens to avoid KeyError during training
                    continue
    return triples


# ----------------------- Logging -----------------------

def _reset_root_logger():
    """Remove all handlers to avoid duplicate log lines when set_logger* is called multiple times."""
    root = logging.getLogger('')
    for h in list(root.handlers):
        root.removeHandler(h)


def set_logger(args):
    """
    Write logs to <save_path>/train.log or test.log and to console.
    Safe to call multiple times (no duplicate handlers).
    """
    _ensure_dir(args.save_path or args.init_checkpoint or '.')
    log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log' if args.do_train else 'test.log')

    _reset_root_logger()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG if getattr(args, 'debug', False) else logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if getattr(args, 'debug', False) else logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def set_logger_(args, detect=True):
    """
    Separate logger used by detect/repair scripts.
    """
    base = getattr(args, 'init_path', None) or getattr(args, 'save_path', None) or '.'
    _ensure_dir(base)
    log_file = os.path.join(base, 'detect_error.log' if detect else 'repair_error.log')

    _reset_root_logger()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# ----------------------- Metrics / misc -----------------------

def is_better_metric(best_metrics, cur_metrics):
    """
    Compare two metrics lists (like those returned by test_step) by overall MRR.
    """
    if best_metrics is None:
        return True
    try:
        return best_metrics[-1]['MRR'] < cur_metrics[-1]['MRR']
    except Exception:
        return True


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs.
    """
    for metric in metrics:
        if 'name' in metric:
            logging.info("results from %s", metric['name'])
        for key in [x for x in metric if x != "name"]:
            try:
                logging.info('%s %s at step %d: %f', mode, key, step, float(metric[key]))
            except Exception:
                logging.info('%s %s at step %d: %s', mode, key, step, str(metric[key]))


def worker_init(worker_id: int):
    """
    Worker seed initializer for PyTorch DataLoader to ensure deterministic numpy/random in each worker.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def binary_cross_entropy(r, x):
    """
    Numerically stable BCE used by VAE parts. Expects r, x on same device.
    """
    eps = 1e-8
    return -torch.sum(x * torch.log(r + eps) + (1 - x) * torch.log(1 - r + eps), dim=-1)


def RotatE_Trans(ent, rel, is_hr):
    """
    Helper for RotatE model; multiplies complex embeddings.
    """
    re_ent, im_ent = ent
    re_rel, im_rel = rel
    if is_hr:  # ent == head
        re = re_ent * re_rel - im_ent * im_rel
        im = re_ent * im_rel + im_ent * re_rel
    else:      # ent == tail
        re = re_rel * re_ent + im_rel * im_ent
        im = re_rel * im_ent - im_rel * re_ent
    return re, im
