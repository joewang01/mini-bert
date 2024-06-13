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
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from tokenizer import BertTokenizer
from transformers import BertTokenizerFast
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
    loadSemanticSimilarityData,
    loadSentimentAnalysisData,
    loadQuoraData
)

from evaluation import model_eval_sst, model_eval_quora, model_eval_sts, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
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


class FurtherPreBert(nn.Module):
    '''
    This module further pretrain BERT with MLM on the specified data set
    '''
    def __init__(self, config):
        super(FurtherPreBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT parameters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        # Linear predictor of masked token
        self.token_prediction = torch.nn.Linear(config.hidden_size, self.tokenizer.vocab_size)



    def forward(self, src_ids, src_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        my_bert = self.bert(src_ids, attention_mask=src_mask)
        return self.token_prediction(src_ids, src_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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

SentimentAnalysisData = {}
QuoraData = {}
SemanticSimilarityData = {}

def loadSentimentAnalysis():
    train, num_labels = loadSentimentAnalysisData(args.sst_train, 'train')
    dev, num_labels = loadSentimentAnalysisData(args.sst_dev, 'train')

    train = SentenceClassificationDataset(train, args)
    dev = SentenceClassificationDataset(dev, args)

    train_dataloader = DataLoader(train, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=train.collate_fn)
    dev_dataloader = DataLoader(dev, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=dev.collate_fn)


    SentimentAnalysisData['train'] = train
    SentimentAnalysisData['dev'] = dev
    SentimentAnalysisData['train_loader'] = train_dataloader
    SentimentAnalysisData['dev_loader'] = dev_dataloader
    SentimentAnalysisData['labels'] = num_labels
    return SentimentAnalysisData
def loadQuora():
    train = loadQuoraData(args.para_train, 'train')
    dev = loadQuoraData(args.para_dev, 'train')

    train = SentencePairDataset(train, args)
    dev = SentencePairDataset(dev, args)

    train_dataloader = DataLoader(train, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=train.collate_fn)
    dev_dataloader = DataLoader(dev, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=dev.collate_fn)


    QuoraData['train'] = train
    QuoraData['dev'] = dev
    QuoraData['train_loader'] = train_dataloader
    QuoraData['dev_loader'] = dev_dataloader

    return QuoraData

def loadSemanticSimilarity():
    train = loadSemanticSimilarityData(args.sts_train, 'train')
    dev = loadSemanticSimilarityData(args.sts_dev, 'train')

    train = SentencePairDataset(train, args, isRegression=True)
    dev = SentencePairDataset(dev, args, isRegression=True)

    train_dataloader = DataLoader(train, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=train.collate_fn)
    dev_dataloader = DataLoader(dev, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=dev.collate_fn)


    SemanticSimilarityData['train'] = train
    SemanticSimilarityData['dev'] = dev
    SemanticSimilarityData['train_loader'] = train_dataloader
    SemanticSimilarityData['dev_loader'] = dev_dataloader

    return SemanticSimilarityData
def train_MLM(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    if args.task == 'sst':
        SentimentAnalysisData = loadSentimentAnalysis()
    if args.task == 'qqq':
        QuoraData = loadQuora()
    if args.task == 'sts':
        SemanticSimilarity = loadSemanticSimilarity()

    # Num labels appears to only be used for SentimentAnalysisData
    num_labels = None
    if args.task == 'sst':
        num_labels = SentimentAnalysisData['labels']
    else:
        num_labels = {}

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'task': args.task}

    config = SimpleNamespace(**config)

    model = FurtherPreBert(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    if args.task == 'sst':
        # Run for the specified number of epochs.
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in tqdm(SentimentAnalysisData['train_loader'], desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = batch['token_ids'], batch['attention_mask'], batch['labels']
                b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

                optimizer.zero_grad()
                seq_ids = b_ids.clone()
                src_mask = generate_square_subsequent_mask(b_ids.size(1))
                rand_value = torch.rand(b_ids.shape)
                rand_mask = (rand_value < 0.15) * (input != 101) * (input != 102) * (input != 0)
                mask_idx = (rand_mask.flatten() == True).nonzero().view(-1)
                seq_ids = seq_ids.flatten()
                seq_ids[mask_idx] = 103
                seq_ids = seq_ids.view(b_ids.size())

                out = model.forward(seq_ids, src_mask)
                loss = nn.CrossEntropyLoss(out.view(-1, ntokens), batch['input_ids'].view(-1).to(device))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            train_acc, *_ = model_eval_sst(SentimentAnalysisData['train_loader'], model, device)
            dev_acc, *_ = model_eval_sst(SentimentAnalysisData['dev_loader'], model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)

            print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    elif args.task == 'qqq':
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in tqdm(QuoraData['train_loader'], desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels = batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels']
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().to(device), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            train_acc, *_ = model_eval_quora(QuoraData['train_loader'], model, device)
            dev_acc, *_ = model_eval_quora(QuoraData['dev_loader'], model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)

            print(
                f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    elif args.task == 'sts':
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in tqdm(SemanticSimilarity['train_loader'], desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels = batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels']
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                # Use MSE Loss, standard for regression problems
                loss = F.mse_loss(logits.float().view(-1), labels.float(), reduction='mean')

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            train_corr, *_ = model_eval_sts(SemanticSimilarity['train_loader'], model, device)
            dev_corr, *_ = model_eval_sts(SemanticSimilarity['dev_loader'], model, device)

            if dev_corr > best_dev_acc:
                best_dev_acc = dev_corr
                save_model(model, optimizer, args, config, args.filepath)

            print(
                f"Epoch {epoch}: train loss :: {train_loss :.3f}, train corr :: {train_corr :.3f}, dev corr :: {dev_corr :.3f}")

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, para_test_data, sts_test_data, num_labels = \
            load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

        sst_dev_data,para_dev_data, sts_dev_data, num_labels = \
            load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
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

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--task", type=str, help="learning rate", choices=('sst', 'qqq', 'sts'), default='sst')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_MLM(args)
