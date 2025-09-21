# model.py — CPU/GPU agnostic, int64-safe indices

import logging
import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# (kept; some forks import this symbolically)
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from ote import OTE
from dataloader import TrainDataset, TestDataset
from helper import *  # RotatE_Trans, TopKHeap, worker_init, etc.

pi = 3.14159265358979323846

# Single source of truth for device (used in a few places if needed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, args,
                 double_entity_embedding=False, double_relation_embedding=False,
                 dropout=0, init_embedding=True):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.scale_relation = True

        # used both for init and scoring in TransE/RotatE
        self.gamma = nn.Parameter(torch.tensor([gamma], dtype=torch.float32), requires_grad=False)

        self.test_split_num = 1

        # (9+2)/1000 ~ 0.011, safe small init range
        self.embedding_range = nn.Parameter(torch.tensor([0.01], dtype=torch.float32), requires_grad=False)
        self._aux = {}

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        self.ote = None

        if model_name == 'OTE':
            assert self.entity_dim % args.ote_size == 0
            self.ote = OTE(args.ote_size, args.ote_scale)
            use_scale = self.ote.use_scale
            self.relation_dim = self.entity_dim * (args.ote_size + (1 if use_scale else 0))

            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(self.relation_embedding, a=0.0, b=1.0)
            if use_scale:
                self.relation_embedding.data.view(-1, args.ote_size + 1)[:, -1] = self.ote.scale_init()
            rel_emb_data = self.orth_rel_embedding()
            self.relation_embedding.data.copy_(rel_emb_data.view(nrelation, self.relation_dim))
        else:
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if init_embedding:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                self.entity_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        else:
            self.entity_embedding = None

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        self.dropout = nn.Dropout(dropout) if dropout > 0 else (lambda x: x)

    def orth_rel_embedding(self):
        rel_emb_size = self.relation_embedding.size()
        ote_size = self.ote.num_elem
        scale_dim = 1 if self.ote.use_scale else 0
        rel_embedding = self.relation_embedding.view(-1, ote_size, ote_size + scale_dim)
        rel_embedding = self.ote.orth_embedding(rel_embedding).view(rel_emb_size)
        if rel_embedding is None:
            rel_embedding = self.ote.fix_embedding_rank(
                self.relation_embedding.view(-1, ote_size, ote_size + scale_dim)
            )
            if self.training:
                self.relation_embedding.data.copy_(rel_embedding.view(rel_emb_size))
                rel_embedding = self.relation_embedding.view(-1, ote_size, ote_size + scale_dim)
            rel_embedding = self.ote.orth_embedding(rel_embedding, do_test=False).view(rel_emb_size)
        return rel_embedding

    def cal_embedding(self):
        if self.model_name == 'OTE':
            rel_embedding = self.orth_rel_embedding()
            self._aux['rel_emb'] = rel_embedding
            self._aux['ent_emb'] = self.entity_embedding

    def get_embedding(self):
        if self.model_name == 'OTE':
            return self._aux['rel_emb'], self._aux['ent_emb']
        return self.relation_embedding, self.entity_embedding

    def reset_embedding(self):
        # clear cached aux embeddings except any marked "static"
        for k in list(self._aux.keys()):
            if k != "static":
                self._aux[k] = None

    def get_model_func(self):
        return {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'OTE': self.OTE,
        }

    def forward(self, sample, mode='single'):
        """
        Calculate the score of a batch of triples.
        Ensures all index tensors are int64 (LongTensor) on the embeddings’ device.
        """
        relation_embedding, entity_embedding = self.get_embedding()
        emb_device = relation_embedding.device  # keep everything on same device as embeddings

        if mode in ('single', 'head-single', 'tail-single'):
            # sample: [B, 3]
            idx_h = sample[:, 0].to(emb_device).long()
            idx_r = sample[:, 1].to(emb_device).long()
            idx_t = sample[:, 2].to(emb_device).long()

            head = torch.index_select(entity_embedding, 0, idx_h).unsqueeze(1)
            relation = torch.index_select(relation_embedding, 0, idx_r).unsqueeze(1)
            tail = torch.index_select(entity_embedding, 0, idx_t).unsqueeze(1)

            head_ids = sample[:, 0].unsqueeze(1) if mode == 'head-single' else sample[:, 0]
            tail_ids = sample[:, 2].unsqueeze(1) if mode == 'tail-single' else sample[:, 2]
            self._aux['samples'] = (head_ids, sample[:, 1], tail_ids, mode)

        elif mode == 'head-batch':
            # sample: (tail_part[B,3], head_part[B, Nneg])
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            idx_head = head_part.view(-1).to(emb_device).long()
            idx_rel = tail_part[:, 1].to(emb_device).long()
            idx_tail = tail_part[:, 2].to(emb_device).long()

            head = torch.index_select(entity_embedding, 0, idx_head).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(relation_embedding, 0, idx_rel).unsqueeze(1)
            tail = torch.index_select(entity_embedding, 0, idx_tail).unsqueeze(1)
            self._aux['samples'] = (head_part, tail_part[:, 1], tail_part[:, 2], mode)

        elif mode == 'tail-batch':
            # sample: (head_part[B,3], tail_part[B, Nneg])
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            idx_h = head_part[:, 0].to(emb_device).long()
            idx_r = head_part[:, 1].to(emb_device).long()
            idx_t = tail_part.view(-1).to(emb_device).long()

            head = torch.index_select(entity_embedding, 0, idx_h).unsqueeze(1)
            relation = torch.index_select(relation_embedding, 0, idx_r).unsqueeze(1)
            tail = torch.index_select(entity_embedding, 0, idx_t).view(batch_size, negative_sample_size, -1)
            self._aux['samples'] = (head_part[:, 0], head_part[:, 1], tail_part, mode)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = self.get_model_func()
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    # ----- Scoring functions -----

    def TransE(self, head, relation, tail, mode, topk=False):
        score = head + (relation - tail) if mode == 'head-batch' else (head + relation) - tail
        if topk:
            return score
        if mode == 'detect':
            score = -torch.norm(score, p=1, dim=1)
        else:
            score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode, topk=False):
        score = head * (relation * tail) if mode == 'head-batch' else (head * relation) * tail
        if topk:
            return score
        return score.sum(dim=2)

    def ComplEx(self, head, relation, tail, mode, topk=False):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        if topk:
            return score
        return score.sum(dim=2)

    def RotatE(self, head, relation, tail, mode, topk=False):
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        phase_relation = relation / (self.embedding_range.item() / pi) if self.scale_relation else relation
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score, im_score = RotatE_Trans((re_tail, im_tail), (re_relation, im_relation), False)
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score, im_score = RotatE_Trans((re_head, im_head), (re_relation, im_relation), True)
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0).norm(dim=0)

        if topk:
            return score

        return self.gamma.item() - score.sum(dim=-1)

    def OTE(self, head, relation, tail, mode, topk=False):
        if mode in ("head-batch", 'head-single'):
            relation = self.ote.orth_reverse_mat(relation)
            output = self.ote(tail, relation)
            if topk:
                return output - head
            score = self.ote.score(output - head)
        else:
            output = self.ote(head, relation)
            if topk:
                return output - tail
            score = self.ote.score(output - tail)
        return self.gamma.item() - score

    # ----- Training / evaluation helpers -----

    @staticmethod
    def apply_loss_func(score, loss_func, is_negative_score=False, label_smoothing=0.1):
        if isinstance(loss_func, nn.SoftMarginLoss):
            tgt_val = -1 if is_negative_score else 1
            tgt = torch.empty_like(score).fill_(tgt_val)
            return loss_func(score, tgt)
        elif isinstance(loss_func, nn.BCELoss):
            tgt_val = 0 if is_negative_score else 1
            if label_smoothing > 0:
                tgt_val = tgt_val * (1 - label_smoothing) + 0.0001
            tgt = torch.empty_like(score).fill_(tgt_val)
            return loss_func(score, tgt)
        else:
            return loss_func(-score) if is_negative_score else loss_func(score)

    @staticmethod
    def train_step(model, train_iterator, confidence, loss_func, args, generator=None):
        """
        One optimization step on a minibatch.
        """
        model.train()

        positive_sample, negative_sample, subsampling_weight, mode, idxs = next(train_iterator)

        # Keep everything on the same device as model params
        model_device = next(model.parameters()).device
        confidence_device = confidence.device if isinstance(confidence, torch.Tensor) else model_device

        idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=confidence_device)
        batch_confidence = torch.index_select(confidence, 0, idx_tensor)

        positive_sample = positive_sample.to(model_device)
        if isinstance(negative_sample, tuple):
            negative_sample = [x.to(model_device) for x in negative_sample]
        else:
            negative_sample = negative_sample.to(model_device)
        # subsampling_weight is unused in this loss variant

        if generator is not None:
            positive_sample, negative_sample = generator.generate(
                model, positive_sample, negative_sample, mode,
                train=False, n_sample=args.negative_sample_size // 2
            )

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            negative_score = (
                F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                * model.apply_loss_func(negative_score, loss_func, True, args.label_smoothing)
            ).sum(dim=1)
        else:
            negative_score = model.apply_loss_func(negative_score, loss_func, True, args.label_smoothing).mean(dim=1)

        pmode = "head-single" if mode == "head-batch" else "tail-single"
        positive_score = model(positive_sample, pmode)
        positive_score = model.apply_loss_func(positive_score, loss_func, False, args.label_smoothing).squeeze(dim=1)

        loss_sign = -1.0
        denom = batch_confidence.sum().clamp_min(1e-6)  # avoid div-by-zero
        positive_sample_loss = loss_sign * (batch_confidence * positive_score).sum() / denom
        negative_sample_loss = loss_sign * (batch_confidence * negative_score).sum() / denom
        loss = 0.5 * (positive_sample_loss + negative_sample_loss)

        # ---- FIXED L3 regularization ----
        if args.regularization != 0.0:
            reg_ent = model.entity_embedding.norm(p=3) ** 3
            reg_rel = model.relation_embedding.norm(p=3) ** 3
            regularization = args.regularization * (reg_ent + reg_rel)
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        model.reset_embedding()
        return log

    def split_test(self, sample, mode):
        if self.test_split_num == 1:
            return self(sample, mode)
        p_sample, n_sample = sample
        scores = []
        sub_samples = torch.chunk(n_sample, self.test_split_num, dim=1)
        for n_ss in sub_samples:
            scores.append(self((p_sample, n_ss.contiguous()), mode))
        return torch.cat(scores, dim=1)

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, head_only=False, tail_only=False):
        """
        Evaluate on test set.
        """
        model.eval()
        model.cal_embedding()
        model_device = next(model.parameters()).device

        # OS-safe dataloaders (0 on Windows to avoid hang; 4 elsewhere)
        num_workers = 0 if os.name == "nt" else 4

        test_dataloader_head = DataLoader(
            TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'head-batch'),
            batch_size=args.test_batch_size, num_workers=num_workers,
            worker_init_fn=worker_init, collate_fn=TestDataset.collate_fn
        )
        test_dataloader_tail = DataLoader(
            TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'tail-batch'),
            batch_size=args.test_batch_size, num_workers=num_workers,
            worker_init_fn=worker_init, collate_fn=TestDataset.collate_fn
        )
        if head_only:
            test_dataset_list = [test_dataloader_head]
        elif tail_only:
            test_dataset_list = [test_dataloader_tail]
        else:
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = [[] for _ in test_dataset_list]

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for k, test_dataset in enumerate(test_dataset_list):
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    positive_sample = positive_sample.to(model_device)
                    negative_sample = negative_sample.to(model_device)
                    filter_bias = filter_bias.to(model_device)

                    batch_size = positive_sample.size(0)

                    score = model.split_test((positive_sample, negative_sample), mode)
                    score += filter_bias * (score.max() - score.min())

                    # Explicit sort to avoid exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        ranking = torch.nonzero(argsort[i, :] == positive_arg[i])
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()
                        logs[k].append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })
                        buf = "%s rank %d " % (mode, ranking) + " ".join(("%s" % int(x) for x in positive_sample[i])) + "\t"
                        buf = buf + " ".join(["%d" % x for x in argsort[i][:10]])
                        logging.debug(buf)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = [{} for _ in logs]
            for i, log in enumerate(logs):
                for metric in log[0].keys():
                    metrics[i][metric] = sum([lg[metric] for lg in log]) / max(1, len(log))
                if len(logs) > 1:
                    metrics[i]['name'] = "head-batch" if i == 0 else "tail-batch"
            if len(logs) == 2:
                metrics_all = {}
                log_all = logs[0] + logs[1]
                for metric in log_all[0].keys():
                    metrics_all[metric] = sum([lg[metric] for lg in log_all]) / max(1, len(log_all))
                metrics_all['name'] = "Overall"
                metrics.append(metrics_all)

        model.reset_embedding()
        return metrics

    @staticmethod
    def find_topk_triples_ssvae(model, train_iterator, labelled_iterator, unlabelled_iterator, noise_triples, rate=10):
        """
        Find top-k triples by model score.
        """
        model.eval()
        noise_triples = set(noise_triples)
        k = max(1, len(train_iterator.dataloader_head.dataset.triples) * rate // 100)
        topk_heap = TopKHeap(k)
        all_triples = train_iterator.dataloader_head.dataset.triples
        model_func = model.get_model_func()
        relation_embedding, entity_embedding = model.get_embedding()
        emb_device = entity_embedding.device
        i = 0
        while i < len(all_triples):
            j = min(i + 1024, len(all_triples))
            sample = torch.tensor(all_triples[i: j], dtype=torch.long, device=emb_device)
            h = torch.index_select(entity_embedding, 0, sample[:, 0]).detach()
            r = torch.index_select(relation_embedding, 0, sample[:, 1]).detach()
            t = torch.index_select(entity_embedding, 0, sample[:, 2]).detach()
            s = model_func[model.model_name](h, r, t, 'single', True)
            score = (-torch.norm(s, p=1, dim=1)).view(-1).detach().cpu().tolist()
            for x, triple in enumerate(all_triples[i: j]):
                topk_heap.push((score[x], triple))
            i = j

        topk_list = topk_heap.topk()
        topk_triples = [t for _, t in topk_list] if len(topk_list) > 0 else []

        labelled_iterator.dataloader_head.dataset.triples = topk_triples
        labelled_iterator.dataloader_tail.dataset.triples = list(labelled_iterator.dataloader_head.dataset.triples)
        labelled_iterator.dataloader_head.dataset.len = len(topk_triples)
        labelled_iterator.dataloader_tail.dataset.len = len(topk_triples)

        num_fake = len(set(topk_triples).intersection(noise_triples))
        logging.info('Fake in top k triples %d / %d' % (num_fake, len(topk_triples)))

        unlabelled_triples = list(set(all_triples) - set(topk_triples))
        unlabelled_iterator.dataloader_head.dataset.triples = unlabelled_triples
        unlabelled_iterator.dataloader_tail.dataset.triples = list(unlabelled_iterator.dataloader_head.dataset.triples)
        unlabelled_iterator.dataloader_head.dataset.len = len(unlabelled_triples)
        unlabelled_iterator.dataloader_tail.dataset.len = len(unlabelled_triples)

    @staticmethod
    def error_detect(kge_model, model, test_triples, true_num, use_sigmoid=False):
        """
        Detect error triples using the classifier output on KGE scores.
        """
        kge_model.eval()
        model.eval()
        kge_model_func = kge_model.get_model_func()
        relation_embedding, entity_embedding = kge_model.get_embedding()
        emb_device = entity_embedding.device
        scores = []
        i = 0
        while i < len(test_triples):
            j = min(i + 1024, len(test_triples))
            sample = torch.tensor(test_triples[i: j], dtype=torch.long, device=emb_device)
            h = torch.index_select(entity_embedding, 0, sample[:, 0]).detach()
            r = torch.index_select(relation_embedding, 0, sample[:, 1]).detach()
            t = torch.index_select(entity_embedding, 0, sample[:, 2]).detach()
            s = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()
            s = s.view(sample.size(0), -1)
            c = model.classify(s).detach().view(-1)
            scores.extend([float(x) for x in c])
            i = j

        error_triples = []
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            TP = FN = FP = TN = 0
            for index, score in enumerate(scores):
                if index < true_num:
                    if score >= threshold:
                        TP += 1
                    else:
                        FN += 1
                        if threshold == 0.5:
                            error_triples.append(test_triples[index])
                else:
                    if score < threshold:
                        TN += 1
                        if threshold == 0.5:
                            error_triples.append(test_triples[index])
                    else:
                        FP += 1

            TN_acc = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            logging.info({
                'threshold': threshold,
                'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
                'TN-acc': TN_acc
            })

        return error_triples

    @staticmethod
    def update_confidence(kge_model, model, train_iterator, confidence, soft, true_num, args):
        """
        Update confidence for all train triples using the SSVAE classifier.
        """
        kge_model.eval()
        model.eval()
        all_triples = train_iterator.dataloader_head.dataset.triples
        kge_model_func = kge_model.get_model_func()
        relation_embedding, entity_embedding = kge_model.get_embedding()
        emb_device = entity_embedding.device
        threshold = args.threshold
        i = 0
        while i < len(all_triples):
            j = min(i + 1024, len(all_triples))
            sample = torch.tensor(all_triples[i: j], dtype=torch.long, device=emb_device)
            h = torch.index_select(entity_embedding, 0, sample[:, 0]).detach()
            r = torch.index_select(relation_embedding, 0, sample[:, 1]).detach()
            t = torch.index_select(entity_embedding, 0, sample[:, 2]).detach()
            s = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()
            s = s.view(sample.size(0), -1)
            c = model.classify(s).detach().view(-1)
            if soft:
                confidence[i: j] = c.to(confidence.device)
            else:
                confidence[i: j] = (c >= threshold).type(torch.float32).to(confidence.device) + 1e-5
            i = j
