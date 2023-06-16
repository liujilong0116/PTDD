# 将文本内容转换为对应的分数向量[15,2]，直接处理所有样本，要用到时后续的代码自行调用

from nsp_model import NSPModel
from tools import *
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
'''
it makes me
i feel
i'am feeling
that sounds
i'am very

in summary, the man is 

'good', 'bad'
'happy', 'sad'
'depressed', 'joyful'
'positive', 'negative'
'great', 'terrible'

'google/multiberts-seed_3-step_2000k',
'google/multiberts-seed_3-step_1000k',
'google/bert_uncased_L-12_H-768_A-12',
'google/bert_uncased_L-8_H-768_A-12',
'google/bert_uncased_L-4_H-768_A-12',
'bert-large-cased',
'bert-large-uncased',
'bert-base-cased',
'dmis-lab/biobert-base-cased-v1.1',
'bert-large-uncased-whole-word-masking'
'emilyalsentzer/Bio_ClinicalBERT'
'''
delete_idx = ['451', '458', '480']  # 只有回答没有问题
root_path = os.getcwd()

save_path = os.path.join(root_path, 'result')
if not os.path.exists(save_path):
    os.mkdir(save_path)

model_name_or_path = 'bert-base-uncased'

model = NSPModel(model_name_or_path)

all_file_name = [i.split('.')[0] for i in os.listdir('./data/topic')]

# label_list = [['good', 'bad'], ['happy', 'sad'], ['depressed', 'joyful'], ['positive', 'negative'], ['great', 'terrible']]
label_list = ['good', 'bad']
score_list = [1, -1]

# prompt_list = [
#     ["i am very {}".format(lable) for lable in label_list],
#     ["i am feeling {}".format(lable) for lable in label_list],
#     ["it makes me {}".format(lable) for lable in label_list],
#     ["i feel {}".format(lable) for lable in label_list],
#     ["that sounds {}".format(lable) for lable in label_list],
#     ["i am {}".format(lable) for lable in label_list],
# ]

with_question = True

prompt_list = ["that sounds {}".format(lable) for lable in label_list]

method_name = prompt_list[0] + '_' + label_list[1]

save_path = save_path + '/%s_%s_%s'%(method_name.replace(' ', '_'), model_name_or_path.replace('/', '-'), "with_question" if with_question else "")
if not os.path.exists(save_path):
    os.mkdir(save_path)

for file_name in all_file_name:
    if not file_name in delete_idx:
        sentence_list = []
        predict_result = 0

        texts = generate_input4topic15(file_name + '.csv', with_question)

        output = model.prodict(texts, prompt_list)
        print(output)

        with open(save_path + '/%s.csv' % file_name, 'a+', newline='') as f:
            f_writer = csv.writer(f)
            for topic_result in output:
                f_writer.writerow(topic_result)
            f.close()