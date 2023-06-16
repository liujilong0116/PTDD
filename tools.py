import csv

def generate_input4topic15(file_name, with_question):
    file_path = './data/topic/%s' % file_name
    out_texts = [None] * 15
    last_topic_index = ''
    last_question = ''
    text_temp = ''
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for index, i in enumerate(csv_reader):
            if index > 0:
                if index != 1 and i[4] != last_topic_index:
                    if with_question:
                        out_texts[int(last_topic_index)] = 'Question: ' + last_question + '? Answer: ' + text_temp
                    else:
                        out_texts[int(last_topic_index)] = text_temp
                    text_temp = i[3] + '. '
                    last_topic_index = i[4]
                    last_question = i[2]
                else:
                    if index == 1:
                        last_topic_index = i[4]
                    text_temp += i[3] + '. '
                    last_question = i[2]
        if with_question:
            out_texts[int(last_topic_index)] = 'Question: ' + last_question + '? Answer: ' + text_temp
        else:
            out_texts[int(last_topic_index)] = text_temp
    return out_texts


# 读取各个Topic的分数，将其转为一个[15,1]的向量
def readTopicScore(file_path, root_path='./result/that_sounds_good_bad_bert-base-uncased_with_question/', parameter=1):
    out = []
    with open(root_path + file_path, 'r') as f:
        f_reader = csv.reader(f)
        for row in f_reader:
            out.append((float(row[1]) - float(row[0])) / parameter)
    return out


