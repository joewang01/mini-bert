
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

from torch.utils.data import DataLoader

truncated_data_proportion = 0.005
truncated_data_proportion_qqq = 0.0001

class SentimentAnalysisData:

    def __init__(self, args):
        self.name = 'sst'
        self.dev_limit = 1102
        self.train_limit = 8545

        if args.trunc_data:
            self.dev_limit = int(self.dev_limit * truncated_data_proportion)
            self.train_limit = int(self.train_limit * truncated_data_proportion)

        self.train_raw, self.labels = loadSentimentAnalysisData(args.sst_train, 'train', limit=self.train_limit)
        self.dev_raw, _ = loadSentimentAnalysisData(args.sst_dev, 'dev', limit=self.dev_limit)

        self.train = SentenceClassificationDataset(self.train_raw, args)
        self.dev = SentenceClassificationDataset(self.dev_raw, args)

        self.train_loader = DataLoader(self.train, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=self.train.collate_fn)

        self.dev_loader = DataLoader(self.dev, shuffle=False, batch_size=args.dev_test_batch_size,
                                    collate_fn=self.dev.collate_fn)

        self.iter = iter(self.train_loader)
        self.steps = 0
class QuoraData:

    def __init__(self, args):
        self.name = 'qqq'
        self.dev_limit = 40430
        self.train_limit = 283003

        if args.trunc_data:
            self.dev_limit = int(self.dev_limit * truncated_data_proportion_qqq)
            self.train_limit = int(self.train_limit * truncated_data_proportion_qqq)

        self.train_raw = loadQuoraData(args.para_train, 'train', limit=self.train_limit)
        self.dev_raw = loadQuoraData(args.para_dev, 'dev', limit=self.dev_limit)


        self.train = SentencePairDataset(self.train_raw, args)
        self.dev = SentencePairDataset(self.dev_raw, args)


        self.train_loader = DataLoader(self.train, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=self.train.collate_fn)
        self.dev_loader = DataLoader(self.dev, shuffle=False, batch_size=args.dev_test_batch_size,
                                    collate_fn=self.dev.collate_fn)

        self.iter = iter(self.train_loader)
        self.steps = 0

class SemanticSimilarityData:

    def __init__(self, args):
        self.name = 'sts'
        self.dev_limit = 864
        self.train_limit = 6041

        if args.trunc_data:
            self.dev_limit = int(self.dev_limit * truncated_data_proportion)
            self.train_limit = int(self.train_limit * truncated_data_proportion)

        self.train_raw = loadSemanticSimilarityData(args.sts_train, 'train', self.train_limit)
        self.dev_raw = loadSemanticSimilarityData(args.sts_dev, 'dev', self.dev_limit)

        self.train = SentencePairDataset(self.train_raw, args, isRegression=True)
        self.dev = SentencePairDataset(self.dev_raw, args, isRegression=True)

        self.train_loader = DataLoader(self.train, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=self.train.collate_fn)
        self.dev_loader = DataLoader(self.dev, shuffle=False, batch_size=args.dev_test_batch_size,
                                    collate_fn=self.dev.collate_fn)

        self.iter = iter(self.train_loader)
        self.steps = 0
