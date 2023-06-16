# 将样本按topic转为15个二维向量，用于text2score15

from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

device = torch.device('cuda:0')
torch.set_printoptions(precision=8)   # 设置打印浮点数位数

class NSPModel():
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path)
        self.model = BertForNextSentencePrediction.from_pretrained(self.model_name_or_path) # "bert-base-uncased"
        self.model.to(device)
        self.softmax = torch.nn.Softmax(dim=0)

    def prodict(self, texts, prompts):
        out_list = []
        for text in texts:
            if text:
                score_list = []
                try:
                    for prompt in prompts:
                        encoding = self.tokenizer(text, prompt, return_tensors="pt")
                        encoding.to(device)
                        outputs = self.model(**encoding, labels=torch.LongTensor([1]).to(device))
                        score_list.append(self.softmax(outputs.logits[0])[0].item())   # [True, False]      [0]:probability that sentences are continuous
                    out_list.append(score_list)
                except Exception as e:
                    pass
                continue
            else:
                out_list.append([0, 0])
        return out_list
