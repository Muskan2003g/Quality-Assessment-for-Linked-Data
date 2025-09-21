# run.py  — CPU/GPU-safe training entry for ZHEClean

import os
import sys

# ---- THREAD CAPS (set env BEFORE importing numpy/torch) ----
# keep libraries from oversubscribing cores (nice for laptops/VMs)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
# (optional) if you use OpenBLAS:
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
# (optional) if you use NumExpr:
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

# repo-specific path
sys.path.append('./semi-supervised')

import argparse
import json
import logging
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# ---- DEVICE & WORKERS (CPU/GPU SAFE) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Windows DataLoader workers can hang; 0 is safest there
TORCH_WORKERS = 0 if os.name == "nt" else 4

# cap PyTorch intra-op threads
try:
    torch.set_num_threads(4)
except Exception:
    pass

from model import KGEModel
from dataloader import TrainDataset, BidirectionalOneShotIterator
from helper import *  # override_config, read_triple, log_metrics, save_model, set_logger / worker_init
from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from models import DeepGenerativeModel, AuxiliaryDeepGenerativeModel


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models.',
        usage='run.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', default=True)
    parser.add_argument('--do_valid', default=True)
    parser.add_argument('--do_test', default=True)

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--noise_rate', type=int, default=20,
                        help='Noisy triples ratio of true train triples.')
    parser.add_argument('--mode', type=str, default='soft', choices=['none', 'soft'])
    parser.add_argument('--update_steps', type=int, default=50000, help='update confidence every xx steps.')
    parser.add_argument('--max_rate', type=int, default=0, help='DO NOT MANUALLY SET')

    # VAE
    parser.add_argument('--vae_model', type=str, default='ADM', choices=['DGM', 'ADM'])
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--a_dim', type=int, default=10)
    parser.add_argument('--h_dim', default='[20, 20]')
    parser.add_argument('--ssvae_steps', type=int, default=3000)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--loss_beta', type=float, default=2.0, help='0.1->2.0')
    parser.add_argument('--mc', type=int, default=5)
    parser.add_argument('--iw', type=int, default=5)

    parser.add_argument('--data_name', type=str, default='WN18RR')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('-save', '--save_path', default='./checkpoint/WN18RR', type=str)
    parser.add_argument('--model', default='RotatE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=512, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=6.0, type=float)  # TransE/RotatE
    parser.add_argument('-adv', '--negative_adversarial_sampling', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=0.5, type=float)

    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')

    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--init_embedding', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--label_smoothing', default=0.1, type=float, help="used for bceloss ")

    parser.add_argument('--seed', default=2021, type=int)

    args = parser.parse_args(args)
    return args


def _safe_set_seeds(seed: int):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)


def _read_dict_any_order(path):
    """
    Accepts both formats:
      1) label<TAB>id
      2) id<TAB>label
    Returns: dict[label] = id
    """
    m = {}
    with open(path, encoding='utf-8') as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            if '\t' not in line:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Bad dict line (need 2 columns): {line}")
                a, b = parts
            else:
                a, b = line.split('\t', 1)

            if a.isdigit() and not b.isdigit():
                label = b
                idx = int(a)
            elif b.isdigit() and not a.isdigit():
                label = a
                idx = int(b)
            else:
                try:
                    idx = int(b)
                    label = a
                except Exception:
                    raise ValueError(f"Cannot parse dict line: {line}")
            m[label] = idx
    return m


def main(args):
    _safe_set_seeds(args.seed)

    # ---- Load configs ----
    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[args.data_name]
    args.data_path = config['data_path']

    # Default save path pattern
    if args.data_name == 'NELL27K':
        args.save_path = './checkpoint/{}-{}-{}'.format(args.data_name, args.model, args.mode)
    else:
        args.save_path = './checkpoint/{}-{}-{}-{}'.format(args.data_name, args.model, args.noise_rate, args.mode)

    # Override from config
    args.batch_size = config['batch_size']
    args.negative_sample_size = config['negative_sample_size']
    args.hidden_dim = config['hidden_dim']
    args.learning_rate = config['lr']
    args.gamma = config['gamma']
    args.adversarial_temperature = config['adversarial_temperature']
    args.max_steps = config['max_steps']
    args.update_steps = config['update_steps']
    args.ssvae_steps = config['ssvae_steps']
    if args.model == 'RotatE':
        args.double_entity_embedding = True

    if args.init_checkpoint:
        override_config(args)

    os.makedirs(args.save_path, exist_ok=True)

    # ---- Logger ----
    # Some repos call set_logger_, others set_logger — try set_logger_, fallback to set_logger.
    try:
        set_logger_(args, detect=False)
    except Exception:
        set_logger(args)

    # ---- Dictionaries (robust to order) ----
    entity2id = _read_dict_any_order(os.path.join(args.data_path, 'entities.dict'))
    relation2id = _read_dict_any_order(os.path.join(args.data_path, 'relations.dict'))

    nentity = len(entity2id)
    nrelation = len(relation2id)
    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s', args.model)
    logging.info('Data Path: %s', args.data_path)
    logging.info('#entity: %d', nentity)
    logging.info('#relation: %d', nrelation)

    # ---- Triples ----
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    true_train_triples = copy.deepcopy(train_triples)

    # default: no synthetic noise
    noise_triples = [(-1, -1, -1)]  # harmless dummy

    if args.mode != "none":
        # Only inject noise when we're actually training a detector
        if args.data_name == 'NELL27K':
            noise_path = os.path.join(args.data_path, 'noise.txt')
        else:
            # e.g., dataset/WN18RR(-mini)/noise_20.txt
            noise_path = os.path.join(args.data_path, f'noise_{args.noise_rate}.txt')

        if os.path.exists(noise_path):
            noise_np = np.loadtxt(noise_path, dtype=np.int32)
            # ensure list[tuple[int,int,int]]
            noise_triples = [tuple(map(int, x)) for x in noise_np.tolist()]
            train_triples = train_triples + noise_triples
            logging.info("Loaded noise from %s (%d triples).", noise_path, len(noise_triples))
        else:
            logging.warning("No noise file found at %s; continuing without synthetic noise.", noise_path)

    logging.info('#train: %d', len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d', len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d', len(test_triples))

    all_true_triples = true_train_triples + valid_triples + test_triples

    # ---- Models ----
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        args=args,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )

    if args.vae_model == 'DGM':
        ssvae_model = DeepGenerativeModel([args.hidden_dim, 1, args.z_dim, eval(args.h_dim)])
    else:
        ssvae_model = AuxiliaryDeepGenerativeModel([args.hidden_dim, 1, args.z_dim, args.a_dim, eval(args.h_dim)])

    logging.info('KGEModel Configuration:')
    logging.info(str(kge_model))
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s', name, str(param.size()), str(param.requires_grad))

    logging.info('AuxiliaryDeepGenerativeModel Configuration:' if args.vae_model == 'ADM'
                 else 'DeepGenerativeModel Configuration:')
    logging.info(str(ssvae_model))
    logging.info('Model Parameter Configuration:')
    for name, param in ssvae_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s', name, str(param.size()), str(param.requires_grad))

    # ---- Move to device (CPU/GPU) ----
    kge_model = kge_model.to(device)
    if args.mode == 'soft':
        ssvae_model = ssvae_model.to(device)

    # ---- DataLoaders (Windows/CPU-safe: num_workers=0) ----
    NUM_WORKERS = TORCH_WORKERS

    if args.do_train:
        train_dataset_head = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch')
        train_dataset_tail = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch')
        train_dataloader_head = DataLoader(
            train_dataset_head,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=worker_init,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            train_dataset_tail,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=worker_init,
            collate_fn=TrainDataset.collate_fn
        )
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        if args.mode == 'soft':
            labelled_triples = random.sample(train_triples, max(1, len(train_triples) // 10))
            labelled_dataset_head = TrainDataset(labelled_triples, nentity, nrelation, 1, 'head-batch')
            labelled_dataset_tail = TrainDataset(labelled_triples, nentity, nrelation, 1, 'tail-batch')
            labelled_dataset_head.true_head, labelled_dataset_head.true_tail = train_dataset_head.true_head, train_dataset_head.true_tail
            labelled_dataset_tail.true_head, labelled_dataset_tail.true_tail = train_dataset_tail.true_head, train_dataset_tail.true_tail
            labelled_dataset_head.count = train_dataset_head.count
            labelled_dataset_tail.count = train_dataset_tail.count

            labelled_dataloader_head = DataLoader(
                labelled_dataset_head, batch_size=args.batch_size, shuffle=True,
                num_workers=NUM_WORKERS, worker_init_fn=worker_init, collate_fn=TrainDataset.collate_fn
            )
            labelled_dataloader_tail = DataLoader(
                labelled_dataset_tail, batch_size=args.batch_size, shuffle=True,
                num_workers=NUM_WORKERS, worker_init_fn=worker_init, collate_fn=TrainDataset.collate_fn
            )
            labelled_iterator = BidirectionalOneShotIterator(labelled_dataloader_head, labelled_dataloader_tail)

            unlabelled_triples = list(set(train_triples) - set(labelled_triples))
            unlabelled_dataset_head = TrainDataset(unlabelled_triples, nentity, nrelation, 1, 'head-batch')
            unlabelled_dataset_tail = TrainDataset(unlabelled_triples, nentity, nrelation, 1, 'tail-batch')
            unlabelled_dataset_head.true_head, unlabelled_dataset_head.true_tail = train_dataset_head.true_head, train_dataset_head.true_tail
            unlabelled_dataset_tail.true_head, unlabelled_dataset_tail.true_tail = train_dataset_tail.true_head, train_dataset_tail.true_tail
            unlabelled_dataset_head.count = train_dataset_head.count
            unlabelled_dataset_tail.count = train_dataset_tail.count

            unlabelled_dataloader_head = DataLoader(
                unlabelled_dataset_head, batch_size=args.batch_size, shuffle=True,
                num_workers=NUM_WORKERS, worker_init_fn=worker_init, collate_fn=TrainDataset.collate_fn
            )
            unlabelled_dataloader_tail = DataLoader(
                unlabelled_dataset_tail, batch_size=args.batch_size, shuffle=True,
                num_workers=NUM_WORKERS, worker_init_fn=worker_init, collate_fn=TrainDataset.collate_fn
            )
            unlabelled_iterator = BidirectionalOneShotIterator(unlabelled_dataloader_head, unlabelled_dataloader_tail)

    # ---- Optimizers ----
    current_learning_rate = args.learning_rate
    if args.do_train:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate, weight_decay=args.weight_decay,
        )

        if args.mode == 'soft':
            beta = DeterministicWarmup(n=2 * len(unlabelled_dataloader_head) * 100 if len(train_triples) else 1000)
            sampler = ImportanceWeightedSampler(mc=args.mc, iw=args.iw)
            elbo = SVI(ssvae_model, likelihood=binary_cross_entropy, beta=beta, sampler=sampler)
            optimizerVAE = torch.optim.Adam(ssvae_model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    # ---- Init / checkpoints ----
    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...', args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'), map_location=device)
        init_step = checkpoint['step']
        if 'score_weight' in kge_model.state_dict() and 'score_weight' not in checkpoint['model_state_dict']:
            checkpoint['model_state_dict']['score_weights'] = kge_model.state_dict()['score_weights']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint.get('current_learning_rate', current_learning_rate)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            current_learning_rate = 0
    elif args.init_embedding:
        logging.info('Loading pretrained embedding %s ...', args.init_embedding)
        if kge_model.entity_embedding is not None:
            entity_embedding = np.load(os.path.join(args.init_embedding, 'entity_embedding.npy'))
            relation_embedding = np.load(os.path.join(args.init_embedding, 'relation_embedding.npy'))
            entity_embedding = torch.from_numpy(entity_embedding).to(kge_model.entity_embedding.device)
            relation_embedding = torch.from_numpy(relation_embedding).to(kge_model.relation_embedding.device)
            kge_model.entity_embedding.data[:entity_embedding.size(0)] = entity_embedding
            kge_model.relation_embedding.data[:relation_embedding.size(0)] = relation_embedding
        init_step = 1
        current_learning_rate = 0
    else:
        logging.info('Randomly Initializing %s Model...', args.model)
        init_step = 1

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d', init_step)
    logging.info('learning_rate = %.5f', current_learning_rate)
    logging.info('batch_size = %d', args.batch_size)
    logging.info('hidden_dim = %d', args.hidden_dim)
    logging.info('gamma = %f', args.gamma)
    logging.info('negative_adversarial_sampling = %s', str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f', args.adversarial_temperature)

    # loss_func is passed to model.apply_loss_func (LogSigmoid variant in original code)
    loss_func = nn.LogSigmoid()
    criterion = nn.BCELoss()

    # ---- Training loop ----
    if args.do_train:
        training_logs = []
        soft = (args.mode == 'soft')
        rate = 10  # used in find_topk_triples_ssvae

        confidence = torch.ones(len(train_triples), requires_grad=False, device=device)

        for step in range(init_step, args.max_steps + 1):
            optimizer.zero_grad()
            log = KGEModel.train_step(kge_model, train_iterator, confidence, loss_func, args)
            optimizer.step()

            training_logs.append(log)

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                }
                save_model(kge_model, ssvae_model if soft else None, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([l[metric] for l in training_logs]) / max(1, len(training_logs))
                log_metrics('Training average', step, [metrics])
                training_logs = []

            if args.mode != 'none' and (step % args.update_steps == 0 or step == args.max_steps):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                kge_model.eval()

                relation_embedding, entity_embedding = kge_model.get_embedding()
                # Update splits for ssvae
                KGEModel.find_topk_triples_ssvae(kge_model, train_iterator, labelled_iterator, unlabelled_iterator, noise_triples, rate=rate)

                kge_model_func = kge_model.get_model_func()

                # ---- Train SSVAE ----
                logging.info('Train ssvae...')
                alpha = args.loss_beta * (1 + len(unlabelled_dataloader_head) / max(1, len(labelled_dataloader_head)))
                clf_loss = 0.0
                ssvae_model.train()
                for i in tqdm(range(args.ssvae_steps)):
                    pos, neg, sub_weight, mode_bt, idx = next(labelled_iterator)
                    u_data, u_neg, sub_weight_u, unlabelled_mode, idx_u = next(unlabelled_iterator)

                    pos = pos.to(device).long()          # ensure Long for index_select
                    neg = neg.to(device).long()
                    u_data = u_data.to(device).long()

                    batch_size, negative_sample = neg.size(0), neg.size(1)

                    # pos_data
                    h = torch.index_select(entity_embedding, 0, pos[:, 0])
                    r = torch.index_select(relation_embedding, 0, pos[:, 1])
                    t = torch.index_select(entity_embedding, 0, pos[:, 2])
                    pos_data = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()

                    # neg_data shapes
                    if mode_bt == 'head-batch':
                        h = torch.index_select(entity_embedding, 0, neg.view(-1)).view(batch_size, negative_sample, -1)
                        r = r.unsqueeze(1)
                        t = t.unsqueeze(1)
                    elif mode_bt == 'tail-batch':
                        h = h.unsqueeze(1)
                        r = r.unsqueeze(1)
                        t = torch.index_select(entity_embedding, 0, neg.view(-1)).view(batch_size, negative_sample, -1)

                    neg_data = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()
                    neg_data = neg_data.view(batch_size, -1)
                    x = torch.cat([pos_data, neg_data], dim=0)
                    y = torch.cat([torch.ones(batch_size, device=device),
                                   torch.zeros(batch_size * negative_sample, device=device)], dim=0).view(-1, 1)

                    # unlabelled
                    h = torch.index_select(entity_embedding, 0, u_data[:, 0])
                    r = torch.index_select(relation_embedding, 0, u_data[:, 1])
                    t = torch.index_select(entity_embedding, 0, u_data[:, 2])
                    u = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()

                    # ELBO + classification
                    L = -elbo(x, y)
                    U = -elbo(u)
                    labels = ssvae_model.classify(x)
                    classification_loss = criterion(labels, y)
                    J_alpha = L + alpha * classification_loss + U
                    optimizerVAE.zero_grad()
                    J_alpha.backward()
                    optimizerVAE.step()
                    clf_loss += classification_loss.item()

                    if i % 200 == 0 and i != 0:
                        cur_log = {
                            'kge_step': step,
                            'ssvae_step': i,
                            'mean_classification_loss': clf_loss / 200.0,
                            'cur_loss': float(J_alpha.item())
                        }
                        logging.info(cur_log)
                        clf_loss = 0.0

                if step == args.max_steps:
                    logging.info('Begin detect error...')
                    num_true = len(test_triples)

                    def maybe_detect(fn, save_suffix=""):
                        path = os.path.join(args.data_path, fn)
                        if not os.path.exists(path):
                            logging.warning("Skip error detect: %s not found", path)
                            return
                        tn = np.loadtxt(path, dtype=np.int32)
                        tn = [tuple(x) for x in tn.tolist()]
                        all_test = test_triples + tn
                        error_triples = KGEModel.error_detect(kge_model, ssvae_model, all_test, num_true)
                        out_name = f'error_triples{save_suffix}.txt'
                        np.savetxt(os.path.join(args.save_path, out_name), error_triples, fmt='%d', delimiter='\t')
                        logging.info("Wrote %s", os.path.join(args.save_path, out_name))

                    if args.data_name == 'NELL27K':
                        maybe_detect('test_negative.txt')
                    else:
                        for rate in [10, 20, 30, 40, 50]:
                            maybe_detect(f'test_negative_{rate}.txt', save_suffix=f'_{rate}')

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
        }
        save_model(kge_model, ssvae_model if (args.mode == 'soft') else None, optimizer, save_variable_list, args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Testing ----
    if args.do_test:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info('Evaluating on Test Dataset...')
        metrics = KGEModel.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())
