'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from itertools import cycle
from types import SimpleNamespace
import os
from pathlib import Path
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataset, sampler
from torch import optim

from bert import BertModel, BertLayer
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

from task_sampler import task_sampler
from bert_pal import BertModelPAL
import wandb

from gradient_surgery import PCGrad
import torch.utils.data as data_utils

TQDM_DISABLE=False
datasets = {}

# This is a convenience class of all important objects/constants such as model, optimizer, lr, config, etc.
# Initialized in initialize_model().
class ModelUtils:
    def __init__(self, args):
        self.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

        # Num labels appears to only be used for SentimentAnalysisData
        if (args.task == 'sst' or args.joint) and 'sst' in datasets:
            self.num_labels = datasets['sst'].labels
        else:
            self.num_labels = {}

        self.config = create_config(args, self.num_labels)
        model = MultitaskBERT(self.config)
        
        bert_config = model.bert.config
        bert_config.agg_method = self.config.agg_method
        
        self.model = model.to(self.device)
        self.model = torch.compile(self.model)

        self.lr = args.lr

        if args.use_gs:
            self.optimizer = PCGrad(AdamW(model.parameters(), lr=self.lr))
        elif args.use_sgd:
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        elif args.use_RMSprop:
            self.optimizer = optim.RMSprop(model.parameters(), lr=self.lr, alpha=0.99, eps=1e-08)
        else:
            self.optimizer = AdamW(model.parameters(), lr=self.lr)

        if args.run_tests or args.test_only:
            self.output_dir_path = 'predictions/' + create_output_dir_path(args)

def create_config(args, num_labels):
    basic_config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                    'num_labels': num_labels,
                    'hidden_size': 768,
                    'data_dir': '.',
                    'fine_tune_mode': args.fine_tune_mode,
                    's_b_qqq_cos': args.s_b_qqq_cos,
                    's_b_sts_cos': args.s_b_sts_cos,
                    's_b_qqq_con': args.s_b_qqq_con,
                    's_b_sts_con': args.s_b_sts_con,
                    'agg_method': args.agg_method,
                    'pal': args.pal,
                    'joint': args.joint
                    }
    if args.joint:
        joint_config = {'joint': args.joint,
                        'pal': args.pal,
                        'pal_enc_dim': args.pal_enc_dim,
                        'nb_tasks': 3,
                        'n_steps': args.n_steps}

        return SimpleNamespace(**{**basic_config, **joint_config})
    else:
        return SimpleNamespace(**{**basic_config, **{'task': args.task}})

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.config = config

        if config.pal:
            self.bert = BertModelPAL.from_pretrained('bert-base-uncased')
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT parameters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        self.construct_sst_model(config)
        self.construct_qqq_model(config)
        self.construct_sts_model(config)
        # if args.joint:
        #     self.construct_sst_model(config)
        #     self.construct_qqq_model(config)
        #     self.construct_sts_model(config)
        # else:
        #     if self.config.task == 'sst':
        #         self.construct_sst_model(config)
        #     elif self.config.task == 'qqq':
        #         self.construct_qqq_model(config)
        #     elif self.config.task == 'sts':
        #         self.construct_sts_model(config)

    def construct_sst_model(self, config):
        self.dropout_sentiment = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_sentiment = torch.nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)

    def construct_qqq_model(self, config):
        # Adding layers for paraphrase (binary, i.e. 2 classes)
        if config.s_b_qqq_cos:
            self.dropout_qqq_1 = torch.nn.Dropout(config.hidden_dropout_prob)
            self.dropout_qqq_2 = torch.nn.Dropout(config.hidden_dropout_prob)
            self.linear_qqq_1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_qqq_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        elif config.s_b_qqq_con:
            self.dropout_para = torch.nn.Dropout(config.hidden_dropout_prob)
            self.classifier_para = torch.nn.Linear(3 * config.hidden_size, 1)
        else:
            self.dropout_para = torch.nn.Dropout(config.hidden_dropout_prob)
            self.classifier_para = torch.nn.Linear(config.hidden_size, 1)

    def construct_sts_model(self, config):
        # Adding layers for paraphrase (binary, i.e. 2 classes)
        if config.s_b_sts_cos:
            self.dropout_sts_1 = torch.nn.Dropout(config.hidden_dropout_prob)
            self.dropout_sts_2 = torch.nn.Dropout(config.hidden_dropout_prob)
            self.linear_sts_1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_sts_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        elif config.s_b_sts_con:
            self.dropout_sts = torch.nn.Dropout(config.hidden_dropout_prob)
            self.regression_sts = torch.nn.Linear(3 * config.hidden_size, 1)
        else:
            self.dropout_sts = torch.nn.Dropout(config.hidden_dropout_prob)
            self.regression_sts = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, task):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        if isinstance(self.bert, BertModelPAL):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask, task=task)['pooler_output']
        else:
            return self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']

    # SST: Sentiment Analysis
    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        bert_sentence_emb = self.forward(input_ids, attention_mask, 0)
        out_bert_drop = self.dropout_sentiment(bert_sentence_emb)
        out_bert_logits = self.classifier_sentiment(out_bert_drop)
        return out_bert_logits


    # QQQ: Paraphrase Detection
    def predict_paraphrase(self, input_ids_1, mask_1, input_ids_2, mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        if self.config.s_b_qqq_cos:
            b_s_1 = self.linear_qqq_1(self.dropout_qqq_1(self.forward(input_ids_1, mask_1, 1)))
            b_s_2 = self.linear_qqq_2(self.dropout_qqq_2(self.forward(input_ids_2, mask_2, 1)))
            out_bert_logits = F.cosine_similarity(b_s_1, b_s_2, dim=1)
        elif self.config.s_b_qqq_con:
            b_s_1 = self.forward(input_ids_1, mask_1, 1)
            b_s_2 = self.forward(input_ids_2, mask_2, 1)
            diff = torch.abs(b_s_1 - b_s_2)
            bert_sentence_emb = torch.cat((b_s_1, b_s_2, diff), dim=1)
            bert_sentence_emb = self.dropout_para(bert_sentence_emb)
            out_bert_logits = self.classifier_para(bert_sentence_emb)
        else:
            bert_ids, bert_att_mask = get_bert_input(input_ids_1, input_ids_2, self.tokenizer)
            bert_sentence_emb = self.forward(bert_ids, bert_att_mask, 1)
            bert_sentence_emb = self.dropout_para(bert_sentence_emb)
            out_bert_logits = self.classifier_para(bert_sentence_emb)

        return out_bert_logits

    # STS: Semantic Textual Similarity
    def predict_similarity(self, input_ids_1, mask_1, input_ids_2, mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        b_s_1, b_s_2 = None, None
        if self.config.s_b_sts_cos:
            b_s_1 = self.linear_sts_1(self.dropout_sts_1(self.forward(input_ids_1, mask_1, 2)))
            b_s_2 = self.linear_sts_2(self.dropout_sts_2(self.forward(input_ids_2, mask_2, 2)))
            out_bert_logits = 2.5 * F.cosine_similarity(b_s_1, b_s_2, dim=1) + 2.5
        elif self.config.s_b_sts_con:
            b_s_1 = self.forward(input_ids_1, mask_1, 2)
            b_s_2 = self.forward(input_ids_2, mask_2, 2)
            diff = torch.abs(b_s_1 - b_s_2)
            bert_sentence_emb = torch.cat((b_s_1, b_s_2, diff), dim=1)
            bert_sentence_emb = self.dropout_sts(bert_sentence_emb)
            out_bert_logits = self.regression_sts(bert_sentence_emb)
        else:
            bert_ids, bert_att_mask = get_bert_input(input_ids_1, input_ids_2, self.tokenizer)
            bert_sentence_emb = self.forward(bert_ids, bert_att_mask, 2)
            bert_sentence_emb = self.dropout_sts(bert_sentence_emb)
            out_bert_logits = self.regression_sts(bert_sentence_emb)

        return out_bert_logits, b_s_1, b_s_2

def get_bert_input(input_ids_1, input_ids_2, tokenizer):
    # We take the approach described in:
    # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    # Bert input is [CLS] sentence_1 [SEP] sentence_2 [SEP] [PAD]... max length equal to longest sentence pair in batch

    # New implementation using Bert tokenizer
    s1 = tokenizer.batch_decode(input_ids_1, skip_special_tokens=True)
    s2 = tokenizer.batch_decode(input_ids_2, skip_special_tokens=True)
    combined = [[a, b] for a, b in zip(s1, s2)]
    ecp = tokenizer(combined, padding=True, return_tensors='pt')
    return ecp['input_ids'].to(input_ids_1.device), ecp['attention_mask'].to(input_ids_1.device)

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

# Need to use iterators when multiplexing between different datasets
def get_batch(name):
    ds = datasets[name]
    try:
        return next(ds.iter)
    except StopIteration:
        ds.iter = cycle(ds.train_loader)
        return next(ds.iter)

def process_sts_batch(batch, o: ModelUtils):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels = batch['token_ids_1'], batch['attention_mask_1'], batch[
        'token_ids_2'], batch['attention_mask_2'], batch['labels']
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels = b_ids_1.to(o.device), b_mask_1.to(o.device), b_ids_2.to(
        o.device), b_mask_2.to(o.device), labels.to(o.device)

    o.optimizer.zero_grad()
    logits, b_s_1, b_s_2 = o.model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    loss = F.mse_loss(logits.float().view(-1), labels.float(), reduction='mean')

    return loss


def process_sst_batch(batch, o: ModelUtils):
    b_ids, b_mask, b_labels = batch['token_ids'], batch['attention_mask'], batch['labels']
    b_ids, b_mask, b_labels = b_ids.to(o.device), b_mask.to(o.device), b_labels.to(o.device)

    o.optimizer.zero_grad()
    logits = o.model.predict_sentiment(b_ids, b_mask)
    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
    return loss

def process_qqq_batch(batch, o: ModelUtils):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels = batch['token_ids_1'], batch['attention_mask_1'], batch[
        'token_ids_2'], batch['attention_mask_2'], batch['labels']
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels = b_ids_1.to(o.device), b_mask_1.to(o.device), b_ids_2.to(
        o.device), b_mask_2.to(o.device), labels.to(o.device)

    o.optimizer.zero_grad()
    logits = o.model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().to(o.device),
                                              reduction='sum') / args.batch_size
    return loss

def get_batch_processing_func(name):
    if name == 'qqq':
        return process_qqq_batch
    elif name == 'sst':
        return process_sst_batch
    elif name == 'sts':
        return process_sts_batch
    else:
        raise ValueError(f"Unknown dataset name: {name}")

def train_single_dataset(dataset, o: ModelUtils):
    best_dev_acc = 0
    for epoch in range(args.epochs):
        o.model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(dataset.train_loader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            func = get_batch_processing_func(dataset.name)
            loss = func(batch, o)
            loss.backward()
            o.optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        model_eval_func = eval_func(dataset.name) # returns an evaluation function for a single dataset
        train_acc, *_ = model_eval_func(dataset.train_loader, o.model, o.device)
        dev_acc, *_ = model_eval_func(dataset.dev_loader, o.model, o.device)

        try:
            log_data = {}
            for name, param in o.model.named_parameters():
                if param.grad is not None:
                    log_data[f"{name}.grad"] = wandb.Histogram(param.grad.cpu().numpy())
                log_data[f"{name}.data"] = wandb.Histogram(param.data.cpu().numpy())
            log_data['train-loss'] = train_loss
            log_data['train-acc'] = train_acc
            log_data['dev-acc'] = dev_acc
            wandb.log(log_data)
        except:
            pass

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(o.model, o.optimizer, args, o.config, args.save_path)

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    if args.run_tests:
        test_multitask(args, o)
def load_datasets(args):
    if args.joint:
        sst = SentimentAnalysisData(args)
        qqq = QuoraData(args)
        sts = SemanticSimilarityData(args)
        datasets[sst.name] = sst
        datasets[qqq.name] = qqq
        datasets[sts.name] = sts

    elif args.task == 'sst':
        sst = SentimentAnalysisData(args)
        datasets[sst.name] = sst
    elif args.task == 'qqq':
        qqq = QuoraData(args)
        datasets[qqq.name] = qqq
    elif args.task == 'sts':
        sts = SemanticSimilarityData(args)
        datasets[sts.name] = sts
    else:
        raise ValueError(f"Unknown dataset name: {args.task}")


def train_multitask(args, o: ModelUtils):
    train_single_dataset(datasets[args.task], o)

def train(epoch, o: ModelUtils, args):

    o.model.train()
    num_batches = 0
    sampler = task_sampler(args.task_sampler, datasets, epoch, args.epochs)
    total_losses = {'qqq': 0, 'sst': 0, 'sts': 0}
    num_train_batches_task = {'qqq': 0, 'sst': 0, 'sts': 0}
    print('Total batches per epoch: ', sampler.total_batches_per_epoch)

    for _ in tqdm(range(args.n_steps), desc=f'train-{epoch}', disable=TQDM_DISABLE):
        batch_task = sampler.get_task(num_batches)
        func = get_batch_processing_func(batch_task)
        loss = func(get_batch(batch_task), o)
        loss.backward()
        o.optimizer.step()

        total_losses[batch_task] += loss.item()
        num_train_batches_task[batch_task] += 1
        num_batches += 1

    return total_losses, num_train_batches_task

def update_task_count(sampler, task):
    sampler.total_batches_remaining_per_task[task] -= 1
    if sampler.total_batches_remaining_per_task[task] <= 0:
        sampler.tasks_with_batches_remaining.remove(task)

def train_with_gs(epoch, o: ModelUtils, args):

    o.model.train()
    total_batch_count = 0
    sampler = task_sampler(args.task_sampler, datasets, epoch, args.epochs)
    batches_per_update = args.update_size

    num_steps = int(sampler.total_batches_per_epoch) // int(batches_per_update)

    total_losses = {'qqq': 0, 'sst': 0, 'sts': 0}
    num_train_batches_task = {'qqq': 0, 'sst': 0, 'sts': 0}

    for _ in tqdm(range(num_steps), desc=f'train-{epoch}', disable=TQDM_DISABLE):
        if len(sampler.total_batches_remaining_per_task) <= 0:
            break

        losses_per_update = {'sst': 0, 'qqq': 0, 'sts': 0}
        tasks_per_update = {'qqq': 0, 'sst': 0, 'sts': 0}
        batches_so_far = 0
        #print("Batches remaining per task", sampler.total_batches_remaining_per_task)
        for task in sampler.tasks_with_batches_remaining:
            update_task_count(sampler, task)

            process_batch = get_batch_processing_func(task)
            batch = get_batch(task)
            loss = process_batch(batch, o)
            total_losses[task] += loss.item()

            losses_per_update[task] += loss
            num_train_batches_task[task] += 1
            tasks_per_update[task] += 1
            total_batch_count += 1
            batches_so_far += 1 # This variable is necessary for when we run out of the other datasets.

        if sampler.total_batches_remaining_per_task['qqq'] > 0:
            for _ in range(batches_per_update - batches_so_far):
                task = 'qqq'
                update_task_count(sampler, task)

                process_batch = get_batch_processing_func(task)
                batch = get_batch(task)
                loss = process_batch(batch, o)

                total_losses[task] += loss.item()
                losses_per_update[task] += loss
                num_train_batches_task[task] += 1
                tasks_per_update[task] += 1
                total_batch_count += 1
        non_zero_tasks = list({k: v for (k, v) in tasks_per_update.items() if v != 0}.keys())
        losses = [losses_per_update[task] / tasks_per_update[task] for task in non_zero_tasks]
        o.optimizer.pc_backward(losses)
        o.optimizer.step()

    return total_losses, num_train_batches_task

def train_with_gs_fixed_steps(epoch, o: ModelUtils, args):
    o.model.train()
    total_batch_count = 0
    sampler = task_sampler(args.task_sampler, datasets, epoch, args.epochs)
    total_losses = {'qqq': 0, 'sst': 0, 'sts': 0}
    num_train_batches_task = {'qqq': 0, 'sst': 0, 'sts': 0}
    batch_size = args.batch_size
    update_size = args.update_size
    # if args.task_sampler == 'annealed':
    #     p_sst = sampler.prob_annealed('sst')
    #     p_qqq = sampler.prob_annealed('qqq')
    #     p_sts = sampler.prob_annealed('sts')
    # elif args.task_sampler == 'prop':
    #     p_sst = sampler.prob_prop('sst')
    #     p_qqq = sampler.prob_prop('qqq')
    #     p_sts = sampler.prob_prop('sts')
    #
    # nb_sst = min(round(update_size * p_sst), 1)
    # nb_sts = min(round(update_size * p_sts), 1)
    # nb_qqq = max(update_size - nb_sst - nb_sts, 0)
    #
    # nbs = {'qqq': nb_qqq, 'sst': nb_sst, 'sts': nb_sts}

    for step in tqdm(range(args.n_steps), desc=f'train-{epoch}', disable=TQDM_DISABLE):
        losses_per_update = {'sst': 0, 'qqq': 0, 'sts': 0}
        tasks_per_update = {'qqq': 0, 'sst': 0, 'sts': 0}

        for b in range(update_size):
            batch_task = sampler.get_task(b)
            process_batch = get_batch_processing_func(batch_task)
            batch = get_batch(batch_task)
            loss = process_batch(batch, o)
            total_losses[batch_task] += loss.item()

            losses_per_update[batch_task] += loss
            num_train_batches_task[batch_task] += 1
            tasks_per_update[batch_task] += 1
            total_batch_count += 1

        non_zero_tasks = list({k: v for (k, v) in tasks_per_update.items() if v != 0}.keys())
        losses = [losses_per_update[task] / tasks_per_update[task] for task in non_zero_tasks]
        o.optimizer.pc_backward(losses)
        o.optimizer.step()

    return total_losses, num_train_batches_task


def train_multitask_joint(args, o: ModelUtils):
    '''Train MultitaskBERT.
    Here are we are jointly training all tasks with different multitask models.
    '''
    best_dev_acc = 0
    for epoch in range(args.epochs):
        if args.use_gs:
            print("Using Gradient Surgery")
            # tl, num_train_batches_task = train_with_gs(epoch, o, args)
            tl, num_train_batches_task = train_with_gs_fixed_steps(epoch, o, args)
            print("Batches per task: ", num_train_batches_task)
        else:
            tl, num_train_batches_task = train(epoch, o, args)
            print("Batches per task: ", num_train_batches_task)

        for t in tl.keys():
            tl[t] = tl[t] / num_train_batches_task[t] if num_train_batches_task[t] > 0 else tl[t]

        (sst_d, _, _, qqq_d, _, _, sts_d, _, _) = model_eval_multitask(datasets['sst'].dev_loader,
                                                                       datasets['qqq'].dev_loader,
                                                                       datasets['sts'].dev_loader, o.model, o.device)

        avg_acc = (sst_d + qqq_d + 0.5 + (0.5 * sts_d)) / 3

        try:
            log_data = {}
            for name, param in o.model.named_parameters():
                if param.grad is not None:
                    log_data[f"{name}.grad"] = wandb.Histogram(param.grad.cpu().numpy())
                log_data[f"{name}.data"] = wandb.Histogram(param.data.cpu().numpy())
            log_data['train-loss-sst'] = tl['sst']
            log_data['train-loss-qqq'] = tl['qqq']
            log_data['train-loss-sts'] = tl['sts']
            log_data['avg-dev-acc'] = avg_acc
            log_data['sst-dev-acc'] = sst_d
            log_data['qqq-dev-acc'] = qqq_d
            log_data['sts-dev-acc'] = sts_d
            wandb.log(log_data)
        except:
            pass

        if avg_acc > best_dev_acc:
            best_dev_acc = avg_acc
            save_model(o.model, o.optimizer, args, o.config, args.save_path)

        print(f"Epoch {epoch}: Avg dev :: {avg_acc :.3f}, dev SST acc :: {sst_d :.3f}, dev QQQ acc :: {qqq_d :.3f}, dev STS corr :: {sts_d :.3f}")
        print(f"Epoch {epoch}: SST mean loss :: {tl['sts'] :.3f}, QQQ mean loss :: {tl['qqq'] :.3f}, STS mean loss :: {tl['sts']  :.3f}")

    if args.run_tests:
        test_multitask(args, o)

def load_saved_model(args, o: ModelUtils):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.load_path, map_location=device)
    model = o.model
    model.load_state_dict(saved['model'])
    model.to(device)
    print(f"Loaded model to test from {args.load_path}")
    return model

def test_multitask(args, o: ModelUtils):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():

        model = load_saved_model(args, o)

        sst_test_data, para_test_data, sts_test_data, num_labels = \
            load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

        sst_dev_data,para_dev_data, sts_dev_data, num_labels = \
            load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.dev_test_batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.dev_test_batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.dev_test_batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.dev_test_batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.dev_test_batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.dev_test_batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, o.device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, o.device)

        with open(o.output_dir_path + '/' + args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(o.output_dir_path + '/' + args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(o.output_dir_path + '/' + args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(o.output_dir_path + '/' + args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(o.output_dir_path + '/' + args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(o.output_dir_path + '/' + args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=32)
    parser.add_argument("--dev_test_batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=32)
    parser.add_argument("--update_size",type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--task", type=str, help="specify the head task", choices=('sst', 'qqq', 'sts'), default='sts')
    parser.add_argument("--task_sampler", type=str, help="learning rate", choices=('annealed', 'sqrt', 'random', 'robin', 'prop'), default='annealed')
    parser.add_argument("--pal", type=bool, default=False)
    parser.add_argument("--joint", action='store_true')
    parser.add_argument("--run_tests", action='store_true')
    parser.add_argument("--use_gs", action='store_true')
    parser.add_argument("--use_sgd", action='store_true')
    parser.add_argument("--use_RMSprop", action='store_true')
    parser.add_argument("--trunc_data", action='store_true')
    parser.add_argument("--pal_enc_dim", type=int, default=204)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--s_b_qqq_cos", action='store_true')
    parser.add_argument("--s_b_sts_cos", action='store_true')
    parser.add_argument("--s_b_qqq_con", action='store_true')
    parser.add_argument("--s_b_sts_con", action='store_true')
    parser.add_argument("--agg_method", choices=('cls', 'mean', 'max'), default='cls')
    parser.add_argument("--n_steps", type=int, default=2500)
    parser.add_argument("--use_wandb", action='store_true')

    args = parser.parse_args()
    return args


def set_wandb(mod_utils, args):
    wandb.init(
        # set the wandb project where this run will be logged
        project="cs224n-dfp",

        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": str(args.pal) + "-" + str(args.pal_enc_dim) + "-" + str(args.task_sampler),
            "epochs": 10,
        }
    )

def param_count_grad(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def param_count(model):
    return sum(param.numel() for param in model.parameters())

def create_output_dir_path(args):
    base = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}'
    if args.joint:
        base += '-joint'
    else:
        base += '-' + args.task + '-'
    if args.s_b_qqq_cos:
        base += '-qqq-cos-'
    if args.s_b_qqq_con:
        base += '-qqq-con-'
    if args.s_b_sts_cos:
        base += '-qqq-cos-'
    if args.s_b_sts_con:
        base += '-qqq-con-'
    if args.use_RMSprop:
        base += '-RMSprop'
    if args.use_sgd:
        base += '-sgd'
    if args.use_gs:
        base += '-gs'
    if args.task_sampler:
        base += f'-{args.task_sampler}'
    if args.pal:
        base += '-pal'
    return base

def move_output_file(o):
    shutil.move("output.txt", o.output_dir_path + "/output.txt")


if __name__ == "__main__":
    args = get_args()
    model_utils = ModelUtils(args)

    if args.run_tests or args.test_only:
        dir_path = model_utils.output_dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if args.test_only:
        print("Running TEST mode only")
        test_multitask(args, model_utils)
    else:
        if args.save_path is None:
            args.save_path = 'multitask.pt'
            args.load_path = args.save_path

        seed_everything(args.seed) # Fix the seed for reproducibility.

        # Note that the order of the following lines is important

        load_datasets(args) # Initializes data_sets dict

        print("# of parameters to be learned", param_count_grad(model_utils.model))

        if args.use_wandb:
            try:
                set_wandb(model_utils, args)
            except:
                print("Not using wandb")
        else:
            print("Not using wandb")

        if args.joint:
            print("Running multi task")
            train_multitask_joint(args, model_utils)
        else:
            print("Running single task")
            train_multitask(args, model_utils)

        if args.test_only or args.run_tests:
            move_output_file(model_utils)

        try:
            wandb.finish()
        except:
            pass

