#!/usr/bin/env python3

'''
Multitask BERT evaluation functions.

When training your multitask model, you will find it useful to call
model_eval_multitask to evaluate your model on the 3 tasks' dev sets.
'''

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np


TQDM_DISABLE = False



# Evaluate multitask model on SST only.
def model_eval_sst(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):

        b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
        b_labels = b_labels.flatten().cpu().numpy()

        y_pred.extend(y_hat)
        y_true.extend(b_labels)
        sent_ids.extend(b_sent_ids)

    acc = accuracy_score(y_true, y_pred)
    # acc = np.mean(np.array(y_pred) == np.array(y_true))
    # F1 score not needed at this point
    # f1 = f1_score(y_true, y_pred, average='macro')

    return acc, y_pred, y_true, sent_ids

def model_eval_quora(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels, b_sent_ids = (batch['token_ids_1'], batch['attention_mask_1'],
                                                                    batch['token_ids_2'], batch['attention_mask_2'],
                                                                    batch['labels'], batch['sent_ids'])
        b_ids_1 = b_ids_1.to(device)
        b_ids_2 = b_ids_2.to(device)
        b_mask_1 = b_mask_1.to(device)
        b_mask_2 = b_mask_2.to(device)

        logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        y_hat = logits.detach().sigmoid().round().flatten().cpu().numpy()
        labels = labels.detach().flatten().cpu().numpy()

        y_true.extend(labels)
        y_pred.extend(y_hat)
        sent_ids.extend(b_sent_ids)

    acc = accuracy_score(y_true, y_pred)

    return acc, y_pred, y_true, sent_ids

def model_eval_sts(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels, b_sent_ids = (batch['token_ids_1'],
                                                                             batch['attention_mask_1'],
                                                                             batch['token_ids_2'],
                                                                             batch['attention_mask_2'], batch['labels'],
                                                                             batch['sent_ids'])

        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)

        logits, _, _ = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        y_hat = logits.detach().flatten().cpu().numpy()
        b_labels = b_labels.detach().flatten().cpu().numpy()

        y_pred.extend(y_hat)
        y_true.extend(b_labels)
        sent_ids.extend(b_sent_ids)

    pearson_mat = np.corrcoef(y_pred, y_true)
    sts_corr = pearson_mat[1][0]

    return sts_corr, y_pred, y_true, sent_ids

# Evaluate multitask model on dev sets.
def model_eval_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        # Evaluate sentiment classification.
        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)
            sst_sent_ids.extend(b_sent_ids)

        sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))

        # Evaluate paraphrase detection.
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            para_sent_ids.extend(b_sent_ids)

        paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))

        # Evaluate semantic textual similarity.
        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits, _, _ = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]

        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}')
        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (sentiment_accuracy, sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids)


# Evaluate multitask model on test sets.
def model_eval_test_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        # Evaluate sentiment classification.
        sst_y_pred = []
        sst_sent_ids = []
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

        # Evaluate paraphrase detection.
        para_y_pred = []
        para_sent_ids = []
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)

        # Evaluate semantic textual similarity.
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits, _, _ = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)

        return (sst_y_pred, sst_sent_ids,
                para_y_pred, para_sent_ids,
                sts_y_pred, sts_sent_ids)

def eval_func(name):
    eval_funcs = { 'sst': model_eval_sst, 'qqq': model_eval_quora, 'sts': model_eval_sts }
    return eval_funcs[name]