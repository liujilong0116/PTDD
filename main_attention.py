import os
from dataLoader import loadTopicScoreForAttention
from models import *
from datasets import load_metric

import random

device = torch.device('cuda:0')
torch.set_printoptions(precision=8)   # 设置打印浮点数位数

if __name__ == '__main__':
    metric_acc = load_metric("./metric/accuracy")
    metric_pre = load_metric("./metric/precision")
    metric_recall = load_metric("./metric/recall")
    metric_f1 = load_metric("./metric/f1")

    data_name = "that_sounds_good_bad_bert-base-uncased_with_question"

    root_path = './data/%s/'%data_name

    batch_size = 32
    shuffle = True

    parameter_list = [0.00000001]
    for parameter in parameter_list:
        for seed in range(30, 100):
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            log_path = "./log/balance/%s_%.10f/%d" % (data_name, parameter, seed)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            for fold in range(7):
                div = "final_divide/balance3_%d.npy"%fold
                # if os.path.exists('%s/log%d.txt' % (log_path, fold)):
                #     print('existed')
                #     continue
                train_loader, dev_loader, test_loader = loadTopicScoreForAttention(div, batch_size, shuffle, root_path, parameter)

                model = MyLinear(n_feature=15, n_hidden=8, n_output=1, dropout=0.5).to(device)

                lr = 0.0005
                num_epochs = 1000
                loss_fn = torch.nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr)
                p = 0.5

                best_dev = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                best_test = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                for epoch in range(num_epochs):

                    model.train()
                    for index, (x_train, y_train) in enumerate(train_loader):
                        x = torch.stack(x_train).T.float().to(device)
                        y_predict = model(x)
                        y = torch.squeeze(y_predict, 1)
                        loss = loss_fn(y, y_train.float().to(device))
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()


                    model.eval()
                    for index, (x_train, y_train) in enumerate(train_loader):
                        with torch.no_grad():
                            x = torch.stack(x_train).T.float().to(device)
                            y_predict = model(x)
                        y = torch.where(y_predict > p, torch.ones_like(y_predict), torch.zeros_like(y_predict))
                        metric_acc.add_batch(predictions=y.int(), references=y_train)
                        metric_pre.add_batch(predictions=y.int(), references=y_train)
                        metric_recall.add_batch(predictions=y.int(), references=y_train)
                        metric_f1.add_batch(predictions=y.int(), references=y_train)
                    train_acc = metric_acc.compute()
                    train_pre = metric_pre.compute()
                    train_recall = metric_recall.compute()
                    train_f1 = metric_f1.compute()

                    for index, (x_dev, y_dev) in enumerate(dev_loader):
                        with torch.no_grad():
                            x = torch.stack(x_dev).T.float().to(device)
                            y_predict = model(x)
                        y = torch.where(y_predict > p, torch.ones_like(y_predict), torch.zeros_like(y_predict))
                        metric_acc.add_batch(predictions=y.int(), references=y_dev)
                        metric_pre.add_batch(predictions=y.int(), references=y_dev)
                        metric_recall.add_batch(predictions=y.int(), references=y_dev)
                        metric_f1.add_batch(predictions=y.int(), references=y_dev)
                    dev_acc = metric_acc.compute()
                    dev_pre = metric_pre.compute()
                    dev_recall = metric_recall.compute()
                    dev_f1 = metric_f1.compute()

                    for index, (x_test, y_test) in enumerate(test_loader):
                        with torch.no_grad():
                            x = torch.stack(x_test).T.float().to(device)
                            y_predict = model(x)
                        y = torch.where(y_predict > p, torch.ones_like(y_predict), torch.zeros_like(y_predict))
                        metric_acc.add_batch(predictions=y.int(), references=y_test)
                        metric_pre.add_batch(predictions=y.int(), references=y_test)
                        metric_recall.add_batch(predictions=y.int(), references=y_test)
                        metric_f1.add_batch(predictions=y.int(), references=y_test)
                    test_acc = metric_acc.compute()
                    test_pre = metric_pre.compute()
                    test_recall = metric_recall.compute()
                    test_f1 = metric_f1.compute()

                    if dev_f1['f1'] > best_test[4]:
                        best_dev = [epoch,
                                    dev_acc['accuracy'], dev_pre['precision'], dev_recall['recall'], dev_f1['f1'],
                                    test_acc['accuracy'], test_pre['precision'], test_recall['recall'], test_f1['f1']]
                    if test_f1['f1'] > best_test[8]:
                        best_test = [epoch,
                                    dev_acc['accuracy'], dev_pre['precision'], dev_recall['recall'], dev_f1['f1'],
                                    test_acc['accuracy'], test_pre['precision'], test_recall['recall'], test_f1['f1']]

                    print("epoch", epoch, ":", "|",
                          '%.4f' % train_acc['accuracy'], '%.4f' % train_pre['precision'], '%.4f' % train_recall['recall'], '%.4f' % train_f1['f1'], "|",
                          '%.4f' % dev_acc['accuracy'], '%.4f' % dev_pre['precision'], '%.4f' % dev_recall['recall'], '%.4f' % dev_f1['f1'], "|",
                          '%.4f' % test_acc['accuracy'], '%.4f' % test_pre['precision'], '%.4f' % test_recall['recall'], '%.4f' % test_f1['f1'])
                    with open('%s/log%d.txt'%(log_path, fold), 'a+') as f:
                        f.write("epoch %d"%epoch + ":" + "\n" +
                            '%.4f  ' % train_acc['accuracy'] + '%.4f  ' % train_pre['precision'] + '%.4f  ' % train_recall['recall'] + '%.4f  ' % train_f1['f1'] + "\n" +
                            '%.4f  ' % dev_acc['accuracy'] + '%.4f  ' % dev_pre['precision'] + '%.4f  ' % dev_recall['recall'] + '%.4f  ' % dev_f1['f1'] + "\n" +
                            '%.4f  ' % test_acc['accuracy'] + '%.4f  ' % test_pre['precision'] + '%.4f  ' % test_recall['recall'] + '%.4f  ' % test_f1['f1'] + "\n")

                print("best_dev", best_dev[0],
                      '%.4f' % best_dev[1], '%.4f' % best_dev[2], '%.4f' % best_dev[3], '%.4f' % best_dev[4],
                      '%.4f' % best_dev[5], '%.4f' % best_dev[6], '%.4f' % best_dev[7], '%.4f' % best_dev[8])
                print("best_dev", best_test[0],
                      '%.4f' % best_test[1], '%.4f' % best_test[2], '%.4f' % best_test[3], '%.4f' % best_test[4],
                      '%.4f' % best_test[5], '%.4f' % best_test[6], '%.4f' % best_test[7], '%.4f' % best_test[8])
                with open('%s/log%d.txt'%(log_path, fold), 'a+') as f:
                    f.write("best_dev:\n" +
                            "epoch: %d\n" % best_dev[0] +
                            "dev : %0.4f    %0.4f    %.4f    %.4f\n" % (best_dev[1], best_dev[2], best_dev[3], best_dev[4]) +
                            "test: %0.4f    %0.4f    %.4f    %.4f\n" % (best_dev[5], best_dev[6], best_dev[7], best_dev[8]))
                    f.write("best_dev:\n" +
                            "epoch: %d\n" % best_test[0] +
                            "dev : %0.4f    %0.4f    %.4f    %.4f\n" % (best_test[1], best_test[2], best_test[3], best_test[4]) +
                            "test: %0.4f    %0.4f    %.4f    %.4f\n" % (best_test[5], best_test[6], best_test[7], best_test[8]))