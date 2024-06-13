import random, numpy as np, argparse
from itertools import cycle
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from tokenizer import BertTokenizer
from transformers import BertTokenizerFast
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from data_utils import SentimentAnalysisData, QuoraData, SemanticSimilarityData

from evaluation import model_eval_multitask, model_eval_test_multitask, eval_func

class task_sampler:
# This class manages the multi-task sampling for a given batch
    def __init__(self, method, datasets, epoch, tot_epoch):
        self.datasets = datasets
        self.tasks = list(self.datasets.keys())
        self.tasks_with_batches_remaining = list(self.datasets.keys())
        self.n_task = np.array([float(len(self.datasets[t].train)) for t in self.tasks])
        self.method = method
        self.E = tot_epoch
        self.e = epoch

        num_sst_batches = len(self.datasets['sst'].train_loader)
        num_sts_batches = len(self.datasets['sts'].train_loader)
        num_qqq_batches = len(self.datasets['qqq'].train_loader)

        self.total_batches_remaining_per_task = {
            'sst': num_sst_batches, 'sts': num_sts_batches, 'qqq': num_qqq_batches}
        self.total_batches_per_epoch = num_sst_batches + num_sts_batches + num_qqq_batches

    def get_task(self, batch_number):
        if self.method == 'sqrt':
            return self.sqrt()
        elif self.method == 'annealed':
            return self.annealed()
        elif self.method == 'random':
            return self.random()
        elif self.method == 'prop':
            return self.random()
        elif self.method == 'robin':
            return self.robin(batch_number)

    def prop(self):
        alpha = 1.0
        p = self.n_task ** alpha
        p = p / p.sum()
        return np.random.choice(self.tasks, p=p)


    def sqrt(self):
        alpha = 0.5
        p = self.n_task ** alpha
        p = p / p.sum()
        return np.random.choice(self.tasks, p=p)

    def annealed(self):
        denom = (self.E - 1.0)
        if denom <= 0: # This is in the case where we trucate everything for a quick dev cycle
            denom = 1
        alpha = 1.0 - 0.8 * self.e / denom
        p = self.n_task ** alpha
        p = p / p.sum()
        return np.random.choice(self.tasks, p=p)

    def random(self):
        return np.random.choice(self.tasks)


    def robin(self, batch_number):
        task = self.tasks_with_batches_remaining[(batch_number + 1) % len(self.tasks_with_batches_remaining)]
        self.total_batches_remaining_per_task[task] -= 1
        if self.total_batches_remaining_per_task[task] <= 0:
            self.tasks_with_batches_remaining.remove(task)
        return task

    def prob_annealed(self, task):
        denom = (self.E - 1.0)
        if denom <= 0: # This is in the case where we truncate everything for a quick dev cycle
            denom = 1
        alpha = 1.0 - 0.8 * self.e / denom
        p = self.n_task ** alpha
        p = p / p.sum()
        return p[self.tasks.index(task)]


