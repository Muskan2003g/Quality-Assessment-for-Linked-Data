# repair_error.py — tolerant, CPU/GPU-safe repair for ZHEClean

import argparse
import os
import json
import random
import logging
from pathlib import Path
import re

import numpy as np
import torch
from tqdm import tqdm

from model import KGEModel
from helper import read_triple, set_logger_


# ----------------------- Device & utils -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_set_seeds(seed: int):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(seed)


def _read_dict_any_order(path: Path):
    """
    Accepts both formats:  label<TAB>id  OR  id<TAB>label
    Returns dict[label] = id
    """
    m = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad dict line (need 2 columns): {line}")
            a, b = parts
            if a.isdigit() and not b.isdigit():
                label, idx = b, int(a)
            elif b.isdigit() and not a.isdigit():
                label, idx = a, int(b)
            else:
                try:
                    idx = int(b)
                    label = a
                except Exception:
                    raise ValueError(f"Cannot parse dict line: {line}")
            m[label] = idx
    return m


def _load_triples_auto(path: Path, entity2id, relation2id):
    """
    Load triples from `path`, handling either label format or numeric IDs.
    Returns: list[(h, r, t)] of ints.
    """
    if not path.exists():
        return []
    # peek first non-empty line
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            break
        else:
            return []

    if all(tok.lstrip("-").isdigit() for tok in toks):
        arr = np.loadtxt(str(path), dtype=np.int32)
        if arr.ndim == 1 and arr.size == 3:
            arr = arr.reshape(1, 3)
        return [tuple(map(int, x)) for x in arr.tolist()]
    else:
        return read_triple(str(path), entity2id, relation2id)


# ----------------------- Argparse -----------------------
def parse_args(args=None):
    p = argparse.ArgumentParser(description="Repair erroneous KG triples.")
    p.add_argument("--data_name", type=str, default="WN18RR")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--init_path", default=None, help="Checkpoint dir; defaults to training pattern.")
    p.add_argument("--model", default="TransE", type=str)
    p.add_argument("-d", "--hidden_dim", default=500, type=int)

    p.add_argument("-de", "--double_entity_embedding", action="store_true")
    p.add_argument("-dr", "--double_relation_embedding", action="store_true")

    # kept for backwards compatibility; not required anymore
    p.add_argument("--train_error_rate", default=20, type=int)
    p.add_argument("--alpha", type=float, default=0.9)

    p.add_argument("--nentity", type=int, default=0)
    p.add_argument("--nrelation", type=int, default=0)

    p.add_argument("--seed", default=2021, type=int)
    return p.parse_args(args)


# ----------------------- Discovery helpers -----------------------
_rate_re = re.compile(r"error_triples_(\d+)\.txt$")


def _discover_predicted_files(init_path: Path, data_name: str):
    """
    Return list of tuples: [(pred_path, rate_opt), ...]
    - For NELL27K we look for single 'error_triples.txt' (rate_opt=None)
    - Otherwise we collect any 'error_triples_XX.txt' that exist.
    """
    found = []
    if data_name == "NELL27K":
        p = init_path / "error_triples.txt"
        if p.exists():
            found.append((p, None))
        return found

    for p in init_path.glob("error_triples_*.txt"):
        m = _rate_re.search(p.name)
        if m:
            found.append((p, int(m.group(1))))
    # sort by rate for a nice order
    found.sort(key=lambda x: (999 if x[1] is None else x[1]))
    return found


# ----------------------- Eval helper -----------------------
def evaluate_repair(clean_triples, all_true_triples_set, num_errors, error_index_set):
    TP = 0
    num_update = 0

    for index, cur_triple in enumerate(clean_triples):
        if index in error_index_set:
            num_update += 1
            if cur_triple in all_true_triples_set:
                TP += 1

    precision = TP / num_update if num_update > 0 else 0.0
    recall = TP / num_errors if num_errors > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    logging.info(f"Precision: {precision:.6f}")
    logging.info(f"Recall:    {recall:.6f}")
    logging.info(f"F1 score:  {f1:.6f}")


# ----------------------- Main -----------------------
if __name__ == "__main__":
    args = parse_args()
    _safe_set_seeds(args.seed)

    # Load configs
    configs = json.load(open("configs.json"))
    configs = {c["name"]: c for c in configs}
    cfg = configs[args.data_name]
    args.data_path = cfg["data_path"]
    args.hidden_dim = cfg["hidden_dim"]

    # Default init_path (compatible with your training save pattern)
    if args.init_path is None:
        if args.data_name == "NELL27K":
            args.init_path = f"./checkpoint/{args.data_name}-{args.model}-soft"
        else:
            # older runs saved with -{noise}-soft where this 'noise' matched training rate;
            # but for discovery we don't need it—user can pass a direct path if desired.
            # This default points to the most common current pattern:
            # "./checkpoint/<data>-<model>-<train_error_rate>-soft"
            args.init_path = f"./checkpoint/{args.data_name}-{args.model}-{args.train_error_rate}-soft"

    if args.model == "RotatE":
        args.double_entity_embedding = True

    # Dicts
    entity2id = _read_dict_any_order(Path(args.data_path) / "entities.dict")
    relation2id = _read_dict_any_order(Path(args.data_path) / "relations.dict")
    args.nentity = len(entity2id)
    args.nrelation = len(relation2id)

    set_logger_(args, detect=False)
    logging.info(args)

    init_dir = Path(args.init_path)
    ckpt_path = init_dir / "checkpoint"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {ckpt_path} — please train with --mode soft first."
        )

    # Build model and load weights
    kge_model = KGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=0,
        args=args,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    kge_model.load_state_dict(checkpoint["model_state_dict"])
    kge_model = kge_model.to(device)
    kge_model.eval()
    model_func = kge_model.get_model_func()

    # Load data (IDs)
    train_triples = _load_triples_auto(Path(args.data_path) / "train.txt", entity2id, relation2id)
    valid_triples = _load_triples_auto(Path(args.data_path) / "valid.txt", entity2id, relation2id)
    test_triples = _load_triples_auto(Path(args.data_path) / "test.txt", entity2id, relation2id)
    all_true_triples = set(train_triples + valid_triples + test_triples)

    # Noise used in training (optional)
    noise_candidates = [
        Path(args.data_path) / "noise.txt",
        Path(args.data_path) / f"noise_{args.train_error_rate}.txt",
    ]
    noise_triples = []
    for npth in noise_candidates:
        if npth.exists():
            noise_triples = _load_triples_auto(npth, entity2id, relation2id)
            break
    if not noise_triples:
        logging.info("No noise file found for training set — proceeding with clean train triples only.")
    all_train_triples = train_triples + noise_triples

    # Build adjacency
    in_triples = {}
    out_triples = {}
    for h, r, t in all_train_triples:
        out_triples.setdefault(h, []).append([r, t])
        in_triples.setdefault(t, []).append([h, r])

    # Embeddings
    entity_embedding = (
        kge_model.entity_embedding.weight.detach()
        if hasattr(kge_model.entity_embedding, "weight")
        else kge_model.entity_embedding.detach()
    )
    relation_embedding = (
        kge_model.relation_embedding.weight.detach()
        if hasattr(kge_model.relation_embedding, "weight")
        else kge_model.relation_embedding.detach()
    )
    nent = entity_embedding.size(0)
    nrel = relation_embedding.size(0)

    # Discover predicted error files to process
    pred_list = _discover_predicted_files(init_dir, args.data_name)
    if not pred_list:
        logging.info("No predicted error files found in checkpoint folder — nothing to repair.")
        # exit cleanly without warnings
        raise SystemExit(0)

    for pred_path, rate in pred_list:
        if args.data_name == "NELL27K":
            logging.info("Repairing predicted errors (single file).")
            out_clean_path = init_dir / "clean_triples.txt"
            gt_err_path = Path(args.data_path) / "test_negative.txt"
        else:
            logging.info(f"Repairing predicted errors at rate={rate}%")
            out_clean_path = init_dir / f"clean_triples_{rate}.txt"
            gt_err_path = Path(args.data_path) / f"test_negative_{rate}.txt"

        # Load predicted errors
        error_triples = _load_triples_auto(pred_path, entity2id, relation2id)
        if not error_triples:
            logging.info(f"{pred_path.name} is empty; skipping.")
            continue

        # Optional evaluation (only if ground-truth negatives present)
        gt_err = _load_triples_auto(gt_err_path, entity2id, relation2id)
        if gt_err:
            gt_err_set = set(gt_err)
            num_errors = len(gt_err_set)
        else:
            gt_err_set = set()
            num_errors = 0
            logging.info(f"No ground-truth negatives for rate={rate} — will skip precision/recall.")

        ground_truth_error_triples_index = set()
        use_outer_power = True
        logging.info(f"Using outer power: {use_outer_power}")

        clean_triples = []
        for triple_index, (h, r, t) in enumerate(tqdm(error_triples, disable=len(error_triples) < 32)):
            if (h, r, t) in gt_err_set:
                ground_truth_error_triples_index.add(triple_index)

            # Candidates: replace h, t, or r
            candidate_triples = (
                [(_, r, t) for _ in range(nent)] +
                [(h, r, _) for _ in range(nent)] +
                [(h, _, t) for _ in range(nrel)]
            )

            inner_power = torch.zeros(2 * nent + nrel)
            i = 0
            while i < len(candidate_triples):
                j = min(i + 4096, len(candidate_triples))
                sample = torch.tensor(candidate_triples[i:j], dtype=torch.long, device=device)
                h_emb = torch.index_select(entity_embedding.to(device), 0, sample[:, 0])
                r_emb = torch.index_select(relation_embedding.to(device), 0, sample[:, 1])
                t_emb = torch.index_select(entity_embedding.to(device), 0, sample[:, 2])

                s = model_func[kge_model.model_name](h_emb, r_emb, t_emb, "single", True)
                score = (-torch.norm(s, p=2, dim=1)).view(-1).detach().cpu()
                inner_power[i:j] = torch.sigmoid(score)
                i = j

            # forbid identity replacements
            inner_power[t] = 0.0
            inner_power[h + nent] = 0.0

            all_power = inner_power.clone()

            if use_outer_power:
                k = min(5, inner_power.numel())
                _, outer_idx = torch.topk(inner_power, k=k)
                outer_power = torch.zeros_like(inner_power)

                for c_index in outer_idx.tolist():
                    c_h, c_r, c_t = candidate_triples[c_index]

                    h_emb0 = entity_embedding[c_h].to(device)
                    r_emb0 = relation_embedding[c_r].to(device)
                    t_emb0 = entity_embedding[c_t].to(device)

                    in_score = 0.0
                    out_score = 0.0

                    if c_h in in_triples and len(in_triples[c_h]) > 0:
                        temp_tr = torch.tensor(in_triples[c_h], dtype=torch.long, device=device)
                        in_h = temp_tr[:, 0]
                        in_r = temp_tr[:, 1]
                        in_h_emb = torch.index_select(entity_embedding.to(device), 0, in_h)
                        in_r_emb = torch.index_select(relation_embedding.to(device), 0, in_r)
                        in_t_emb = t_emb0

                        if args.model == "TransE":
                            in_r_emb = in_r_emb + r_emb0
                        elif args.model == "RotatE":
                            in_r_emb = in_r_emb * r_emb0

                        s_in = model_func[kge_model.model_name](in_h_emb, in_r_emb, in_t_emb, "single", True)
                        sc = (-torch.norm(s_in, p=2, dim=-1)).view(-1).detach().cpu()
                        in_score = torch.sigmoid(torch.mean(sc)).item()

                    if c_t in out_triples and len(out_triples[c_t]) > 0:
                        temp_tr = torch.tensor(out_triples[c_t], dtype=torch.long, device=device)
                        out_r = temp_tr[:, 0]
                        out_t = temp_tr[:, 1]
                        out_t_emb = torch.index_select(entity_embedding.to(device), 0, out_t)
                        out_h_emb = h_emb0
                        out_r_emb = torch.index_select(relation_embedding.to(device), 0, out_r)

                        if args.model == "TransE":
                            out_r_emb = out_r_emb + r_emb0
                        elif args.model == "RotatE":
                            out_r_emb = out_r_emb * r_emb0

                        s_out = model_func[kge_model.model_name](out_h_emb, out_r_emb, out_t_emb, "single", True)
                        sc = (-torch.norm(s_out, p=2, dim=-1)).view(-1).detach().cpu()
                        out_score = torch.sigmoid(torch.mean(sc)).item()

                    if in_score == 0.0 and out_score == 0.0:
                        outer_power[c_index] = inner_power[c_index]
                    elif in_score == 0.0:
                        outer_power[c_index] = out_score
                    elif out_score == 0.0:
                        outer_power[c_index] = in_score
                    else:
                        outer_power[c_index] = 0.5 * (in_score + out_score)

                    all_power[c_index] = args.alpha * inner_power[c_index] + (1 - args.alpha) * outer_power[c_index]

            # choose best candidate
            index = torch.argmax(all_power).item()
            if index < nent:
                clean_triples.append((index, r, t))
            elif index < 2 * nent:
                clean_triples.append((h, r, index - nent))
            else:
                clean_triples.append((h, index - 2 * nent, t))

        # Save cleaned triples
        np.savetxt(str(out_clean_path), clean_triples, fmt="%d", delimiter="\t")
        logging.info(f"Saved repaired triples to: {out_clean_path}")

        # Evaluate only if we had GT negatives
        if num_errors > 0:
            evaluate_repair(clean_triples, all_true_triples, num_errors, ground_truth_error_triples_index)

        if args.data_name == "NELL27K":
            break

    logging.info("Repair completed without warnings.")
