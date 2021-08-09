from tqdm import tqdm
import os
import numpy as np
import yaml
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from model import Classifier, FocalLoss
from dataset import get_dataset

from torch.nn.functional import cross_entropy

from torch.optim import AdamW
from torch.utils.data import DataLoader


def train(config):
    # 指定可用GPU数量
    device_ids = [0, 1, 2, 3]
    CURRENT_DIR = config["CURRENT_DIR"]
    logger.info("构建数据集...")
    train_set, eval_set = get_dataset("extreme_unbalance/train.txt"), get_dataset("eval/eval.txt")
    logger.info("加载模型...")
    model = Classifier()
    if config["use_cuda"] and torch.cuda.is_available():
        # model = torch.nn.DataParallel(model)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model = model.cuda()
        model = model.cuda(device=device_ids[0])
        # model = torch.nn.parallel.DistributedDataParallel(model)
    logger.info("加载模型完成...")
    train_dataloader = DataLoader(dataset=train_set, batch_size=config["batch_size"], shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_set, batch_size=config["batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), config["LR"])
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_set))
    logger.info("  Num eval examples = %d", len(eval_set))
    # logger.info("  Num test examples = %d", len(test_dataloader)*config["batch_size"])
    logger.info("  Num Epochs = %d", config["EPOCH"])
    logger.info("  Learning rate = %d", config["LR"])

    model.train()

    for epoch in range(config["EPOCH"]):
        for index, batch in enumerate(train_dataloader):
            # print(batch)
            # break
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids = \
                batch[0].squeeze(), batch[1].squeeze(), batch[2].squeeze()
            label = batch[3]
            if config["use_cuda"] and torch.cuda.is_available():
                input_ids, attention_mask, token_type_ids = \
                    input_ids.cuda(device=device_ids[0]), attention_mask.cuda(
                        device=device_ids[0]), token_type_ids.cuda(device=device_ids[0])
                label = label.cuda(device=device_ids[0])
            model_output = model(input_ids, attention_mask, token_type_ids)
            loss_f = FocalLoss()
            # loss_f = cross_entropy
            train_loss = loss_f(model_output, label)
            train_loss.backward()
            optimizer.step()

            if index % 10 == 0 and index > 0:
                logger.info("train epoch {}/{} batch {}/{} loss {}".format(str(epoch), str(config["EPOCH"]), str(index),
                                                                           str(len(train_dataloader)),
                                                                           str(train_loss.item())))
            if index % 20 == 0 and index > 0:
                evaluate(config, model, eval_dataloader, device_ids)
                if index > 0:
                    checkpoint_name = os.path.join(config["checkpoint_path"],
                                                   "checkpoint-epoch{}-batch{}.bin".format(str(epoch), str(index)))
                    torch.save(model.state_dict(), checkpoint_name)
                    logger.info("saved model!")
            model = model.train()
        # logger.info("test！！！")
        # evaluate(config, model, test_dataloader, device_ids)


def evaluate(config, model, eval_dataloader, device_ids):
    # test
    model = model.eval()
    logger.info("eval!")
    loss_sum = 0

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    # 创建混淆矩阵
    cls_nums = len(config["categories"])
    confuse_matrix = np.zeros((cls_nums, cls_nums))

    for index, batch in enumerate(eval_dataloader):
        # if index % 10 == 0:
        #     print("eval{}/{}".format(str(index), str(len(eval_dataloader))))
        input_ids, attention_mask, token_type_ids = \
            batch[0].squeeze(), batch[1].squeeze(), batch[2].squeeze()
        label = batch[3]
        if config["use_cuda"] and torch.cuda.is_available():
            input_ids, attention_mask, token_type_ids = \
                input_ids.cuda(device=device_ids[0]), attention_mask.cuda(device=device_ids[0]), token_type_ids.cuda(
                    device=device_ids[0])
            label = label.cuda(device=device_ids[0])
        model_output = model(input_ids, attention_mask, token_type_ids)
        loss_f = FocalLoss()
        # loss_f = cross_entropy
        eval_loss = loss_f(model_output, label)
        loss_sum = loss_sum + eval_loss.item()

        pred = torch.argmax(model_output, dim=1)

        correct += (pred == label).sum().float()
        total += len(label)
        for index in range(len(pred)):
            confuse_matrix[label[index]][pred[index]] = confuse_matrix[label[index]][pred[index]] + 1

    logger.info("eval loss: {}".format(str(loss_sum / (len(eval_dataloader)))))
    logger.info("eval accu: {}".format(str((correct / total).cpu().detach().data.numpy())))
    logger.info("confuse_matrix:")
    for i in range(cls_nums):
        strs = config["categories"][i]
        for j in range(cls_nums):
            strs = strs + str(confuse_matrix[i][j]) + " |"
        logger.info(strs)

    for i in range(cls_nums):
        strs = config["categories"][i]
        p, r = 0, 0
        for j in range(cls_nums):
            p = p + confuse_matrix[j][i]
            r = r + confuse_matrix[i][j]
        strs = strs + " 精度 {}".format(str(confuse_matrix[i][i] / p)) + " 召回率 {}".format(str(confuse_matrix[i][i] / r))
        logger.info(strs)


if __name__ == '__main__':
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    train(config)