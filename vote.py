import torch

from dataLoader import loadTopicScoreForVote
from datasets import load_metric

if __name__ == '__main__':
    metric_acc = load_metric("./metric/accuracy")
    metric_pre = load_metric("./metric/precision")
    metric_recall = load_metric("./metric/recall")
    metric_f1 = load_metric("./metric/f1")

    root_path = './result/that_sounds_good_bad_bert-base-uncased_with_question/'
    batch_size = 256
    shuffle = False
    data_loader = loadTopicScoreForVote(batch_size, shuffle, root_path, parameter=0.00000000000001)


    for index, (x_test, y_test) in enumerate(data_loader):
        y = []
        x = torch.stack(x_test).T.float()
        x = torch.sigmoid(x) - torch.full(x.shape, 0.5)
        for i in range(len(x)):
            out = sum(x[i])
            if not out < 0:
                y.append(1)
            else:
                y.append(0)
        metric_acc.add_batch(predictions=y, references=y_test)
        metric_pre.add_batch(predictions=y, references=y_test)
        metric_recall.add_batch(predictions=y, references=y_test)
        metric_f1.add_batch(predictions=y, references=y_test)
    acc = metric_acc.compute()
    pre = metric_pre.compute()
    recall = metric_recall.compute()
    f1 = metric_f1.compute()
    print("epoch", "init", ":", "|",
          '%.4f' % acc['accuracy'], '%.4f' % pre['precision'], '%.4f' % recall['recall'], '%.4f' % f1['f1'])
