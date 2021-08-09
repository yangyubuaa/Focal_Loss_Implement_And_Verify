import torch
from transformers import AutoModel, AutoTokenizer

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bert_model = AutoModel.from_pretrained("bert-base-chinese")
        self.dense = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_features = self.bert_model(input_ids, attention_mask, token_type_ids)
        last_hidden_state = bert_features.last_hidden_state
        return self.dense(last_hidden_state[:, 0, :])


class FocalLoss(torch.nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, model_output, ground_truth):
        ground_truth = ground_truth.unsqueeze(-1)
        normalized = self.softmax(model_output)
        result = normalized.gather(1, ground_truth)
        tolog = torch.log(result)
        loss = - (1 - result) * tolog
        return torch.sum(loss, dim=0) / loss.shape[0]
        




if __name__ == '__main__':
    c = Classifier()
