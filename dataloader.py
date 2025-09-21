# dataloader.py — CPU/GPU safe, int64 indices everywhere

import numpy as np
import torch
import collections
from torch.utils.data import Dataset


class ClfTrainDataset(Dataset):
    def __init__(self, triples, labels):
        self.len = len(triples)
        self.triples = triples
        self.labels = labels

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.triples[idx]
        label = self.labels[idx]
        return sample, label


class TrainDataset(Dataset):
    def __init__(
        self,
        triples,
        nentity,
        nrelation,
        negative_sample_size,
        mode,
        negative_sample_head_size=1,
        negative_sample_tail_size=2,
        half_correct=False
    ):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.count_rels = self.count_rel_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.rel_tail = self.get_true_relation_to_tail(self.triples)

        # only valid for rel-batch mode
        self.negative_sample_head_size = negative_sample_head_size
        self.negative_sample_tail_size = negative_sample_tail_size
        if mode == 'rel-batch':
            assert self.negative_sample_head_size >= 1
            assert self.negative_sample_tail_size >= 1
        self.half_correct = half_correct

    def __len__(self):
        return self.len

    def sample_negative_sample(self, sample_size, head, rel, tail, mode):
        """
        Return numpy array (int64) of size [sample_size] with entity IDs *not forming a true triple*
        for the given (h, r, t, mode).
        """
        negative_sample_size = 0
        negative_sample_list = []
        while negative_sample_size < sample_size:
            negative_sample = np.random.randint(self.nentity, size=sample_size * 2, dtype=np.int64)
            if mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(rel, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, rel)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:sample_size].astype(np.int64, copy=False)
        return negative_sample

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        positive_sample = torch.tensor([head, relation, tail], dtype=torch.long)

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1.0 / torch.tensor([subsampling_weight], dtype=torch.float32))

        if self.mode == 'rel-batch':
            # Not used in your current run pipeline, but keep it correct and int64-safe.
            subsampling_weight = self.count_rels[relation]
            subsampling_weight = torch.sqrt(1.0 / torch.tensor([subsampling_weight], dtype=torch.float32))

            # Sample different heads (ensuring not equal to the positive head)
            # Start with a pool and filter; keep trying until enough samples collected.
            negative_sample_head = np.empty((0,), dtype=np.int64)
            while negative_sample_head.size < self.negative_sample_head_size:
                cand = np.random.randint(self.nentity, size=self.negative_sample_head_size * 5, dtype=np.int64)
                cand = cand[cand != head]
                negative_sample_head = np.unique(np.concatenate([negative_sample_head, cand]))
                if negative_sample_head.size > self.negative_sample_head_size:
                    negative_sample_head = negative_sample_head[:self.negative_sample_head_size]
                    break
            # ensure at least one (fallback)
            if negative_sample_head.size == 0:
                negative_sample_head = np.array([head], dtype=np.int64)

            negative_sample_tails = []
            for i in range(self.negative_sample_head_size):
                neg_t = self.sample_negative_sample(
                    self.negative_sample_tail_size,
                    int(negative_sample_head[i]), relation, -1,
                    'tail-batch'
                )
                negative_sample_tails.append(neg_t.astype(np.int64, copy=False))
            negative_sample_tail = np.stack(negative_sample_tails, axis=0)

            negative_sample = (
                torch.from_numpy(negative_sample_head.astype(np.int64, copy=False)),
                torch.from_numpy(negative_sample_tail.astype(np.int64, copy=False)),
            )
        else:
            negative_sample_np = self.sample_negative_sample(
                self.negative_sample_size,
                head, relation, tail,
                self.mode
            )
            negative_sample = torch.from_numpy(negative_sample_np.astype(np.int64, copy=False))

        return positive_sample, negative_sample, subsampling_weight, self.mode, idx

    @staticmethod
    def collate_fn(data):
        """
        Batches:
          - positive_sample: [B, 3] long
          - negative_sample: [B, N] long  OR tuple(head:[B,H], tail:[B,H,T]) long
          - subsample_weight: [B] float
          - mode: str
          - idxs: list[int]
        """
        positive_sample = torch.stack([_[0] for _ in data], dim=0).long()

        if isinstance(data[0][1], tuple):
            negative_sample_head = torch.stack([_[1][0] for _ in data], dim=0).long()
            negative_sample_tail = torch.stack([_[1][1] for _ in data], dim=0).long()
            negative_sample = (negative_sample_head, negative_sample_tail)
        else:
            negative_sample = torch.stack([_[1] for _ in data], dim=0).long()

        subsample_weight = torch.cat([_[2] for _ in data], dim=0).float()
        idxs = [int(_[4]) for _ in data]
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode, idxs

    @staticmethod
    def count_frequency(triples, start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        """
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def count_rel_frequency(triples):
        relations = [x[1] for x in triples]
        counter = collections.Counter(relations)
        return counter

    @staticmethod
    def get_true_head_and_tail(triples):
        """
        Build dicts of true heads/tails (numpy int64 arrays) for filtering in negative sampling.
        """
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in list(true_head.keys()):
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])), dtype=np.int64)
        for head, relation in list(true_tail.keys()):
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])), dtype=np.int64)

        return true_head, true_tail

    @staticmethod
    def get_true_relation_to_tail(triples):
        """
        Build a dictionary of true tails given the relation (numpy int64 arrays)
        """
        tail_map = {}
        for _, relation, tail in triples:
            if relation not in tail_map:
                tail_map[relation] = [tail]
            else:
                tail_map[relation].append(tail)
        for rel in list(tail_map.keys()):
            tail_map[rel] = np.array(list(set(tail_map[rel])), dtype=np.int64)
        return tail_map


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        # tmp -> tensor
        tmp_t = torch.tensor(tmp, dtype=torch.long)  # [nentity, 2]
        filter_bias = tmp_t[:, 0].float()
        negative_sample = tmp_t[:, 1].long()

        positive_sample = torch.tensor((head, relation, tail), dtype=torch.long)

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0).long()
        negative_sample = torch.stack([_[1] for _ in data], dim=0).long()
        filter_bias = torch.stack([_[2] for _ in data], dim=0).float()
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.dataloader_head = dataloader_head
        self.dataloader_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(self.dataloader_head)
        self.iterator_tail = self.one_shot_iterator(self.dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        Transform a PyTorch Dataloader into python iterator
        """
        while True:
            for data in dataloader:
                yield data
