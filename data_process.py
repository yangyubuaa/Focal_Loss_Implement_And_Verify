if __name__ == '__main__':
    with open("train.txt", "r", encoding="utf-8") as train_r:
        datas = [i.strip() for i in train_r.readlines()]
    pos = list()
    neg = list()
    for i in datas:
        if i[-1] == "1":
            pos.append(i)
        else:
            neg.append(i)
    print(len(pos), len(neg), len(datas))
    # pos 146502 neg 135385 datas 281887

    # 构建测试集，平衡
    eval = [i for i in pos[:5000]]
    for j in neg[:5000]:
        eval.append(j)

    with open("eval/eval.txt", "w", encoding="utf-8") as balance_w:
        for i in eval:
            balance_w.write(i+"\n")

    pos = pos[5000:]
    neg = neg[5000:]
    print(len(eval), len(pos), len(neg))

    pos_balance = [i for i in pos[:100000]]
    neg_balance = [i for i in neg[:100000]]

    import random
    balance_dataset = pos_balance + neg_balance
    random.shuffle(balance_dataset)

    with open("balance/train.txt", "w", encoding="utf-8") as balance_w:
        for i in balance_dataset:
            balance_w.write(i+"\n")

    pos_unbalance = [i for i in pos[:130000]]
    neg_unbalance = [i for i in neg[:60000]]
    unbalance_dataset = pos_unbalance + neg_unbalance
    random.shuffle(unbalance_dataset)
    with open("unbalance/train.txt", "w", encoding="utf-8") as balance_w:
        for i in unbalance_dataset:
            balance_w.write(i+"\n")

    pos_ext = [i for i in pos[:130000]]
    neg_ext = [i for i in neg[:13000]]
    extreme = pos_ext + neg_ext
    random.shuffle(extreme)
    with open("extreme_unbalance/train.txt", "w", encoding="utf-8") as unbalance_w:
        for i in extreme:
            unbalance_w.write(i+"\n")
