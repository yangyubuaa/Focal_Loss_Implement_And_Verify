import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, inputs, labels):
        self.input_ids = inputs[0]
        self.attention_mask = inputs[1]
        self.token_type_ids = inputs[2]
        self.labels = labels

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item], self.token_type_ids[item], torch.tensor(int(self.labels[item]))

    def __len__(self):
        return len(self.input_ids)


def get_dataset(path):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    with open(path, "r", encoding="utf-8") as train_r:
        train_datas = [i.strip() for i in train_r.readlines()]

    inputs = [list(), list(), list()]
    labels = list()
    for i in tqdm(train_datas):
        i_split = i.split("\t")
        tokenized = tokenizer(i_split[0], return_tensors="pt", max_length=20, truncation=True, padding="max_length")
        inputs[0].append(tokenized["input_ids"])
        inputs[1].append(tokenized["attention_mask"])
        inputs[2].append(tokenized["token_type_ids"])
        labels.append(i_split[1])

    return TestDataset(inputs, labels)
