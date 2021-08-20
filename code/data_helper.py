# -*- coding: UTF-8 -*-
from bert import tokenization
import os
import copy
import json
import random
from itertools import chain
import pandas as pd
import sys
import re
from TurkishStemmer import TurkishStemmer
sys.path.append(os.path.dirname(os.getcwd()))

turkStem = TurkishStemmer()


class TrainData(object):
    def __init__(self, config):

        self.__vocab_path = os.path.join(
            config["bert_model_path"], "vocab.txt")
        self.__output_path = config["output_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)
        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长
        self._batch_size = config["batch_size"]

        self.__num_samples = config["num_samples"]

        self.count = 0

    @staticmethod
    def load_data(train_file, eval_file):
        """
        :return: DataFrame for all query both in English and Turkish
                 train DataFrame
                 eval DataFrame
        """
        
        train_data = pd.read_csv(train_file)
        train_data = train_data[train_data.link_index != '(-1, -1)'].reset_index(drop = True)

        eval_data = pd.read_csv(eval_file)
        eval_data = eval_data[eval_data.link_index != '(-1, -1)'].reset_index(drop = True)

        return train_data, eval_data

    def sentence_process(self, sentence):
        replacement_pool = [
            ['<br>', ' '],
            ['"', ' '],
            ['\'', ' '],
            ['.', ' '],
            [',', ' '],
            ['?', ' '],
            ['!', ' '],
            ['[', ' '],
            [']', ' '],
            ['(', ' '],
            [')', ' '],
            ['{', ' '],
            ['}', ' '],
            ['<', ' '],
            ['>', ' '],
            [':', ' '],
            ['\\', ' '],
            ['`', ' '],
            ['=', ' '],
            ['$', ' '],
            ['/', ' '],
            ['*', ' '],
            [';', ' '],
            ['-', ' '],
            ['^', ' '],
            ['|', ' '],
            ['%', ' '],
            ['\/', ' '],
        ]
        sentence = sentence.lower()
        for rule in replacement_pool:
            sentence = sentence.replace(rule[0], rule[1])
        
        return sentence    

    def get_sample(self, queries, query, ranking, sampled_queries, qids):
        link_index = queries[(queries.loc[:, 'query'] == query) & (
            queries.ranking == ranking)].reset_index().at[0, 'link_index'][1:-1]
        qid = qids[sampled_queries.index(query)]
        part = link_index.split(',')[0]
        row = int(link_index.split(',')[1])
        lang = qid[0:2]
        
        if lang == 'en':
            with open('data/en_list_result/part-%s' % part.zfill(5), 'r') as fp:
                line = fp.readlines()[row]
        elif lang == 'tr':
            with open('data/tr_list_result/part-%s' % part.zfill(5), 'r') as fp:
                line = fp.readlines()[row]
        else:
            print("the name of qid Error\n")
            line = "Error"

        title = line.split('\x01')[1] * 20  # title has greater weight
        content = line.split('\x01')[2]
        link = line.split('\x01')[0]
        sample = self.sentence_process(title + content) + self.getWordsFromURL(link)

        return sample

    def getWordsFromURL(self, url):
        words_list = re.compile(r'[\:/?=\-&.,_@%!$0123456789()&*+\[\]]+',re.UNICODE).split(url)
        drop_words = set(['', 'http', 'https', 'www', 'com', '\t', 'm', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'])
        url_words = [turkStem.stem(word.lower()) for word in words_list if word.lower() not in drop_words]
        return " ".join(url_words)



    def neg_samples(self, queries, n_tasks):
        """
        随机负采样多个样本
        :param queries: all of the queries data
        :param n_tasks: set manualy 
        :return: [], [[]]
        """
        '''
        new_queries = []
        new_sims = []

        for i in range(n_tasks):
            questions = random.choice(queries)
            copy_questions = copy.copy(queries)
            copy_questions.remove(questions)
            pos_samples = random.sample(questions, 2)

            copy_questions = list(chain(*copy_questions))
            neg_sims = random.sample(copy_questions, self.__num_samples - 1)
            new_queries.append(pos_samples[0])
            new_sims.append([pos_samples[1]] + neg_sims)
    
        return new_queries, new_sims
        '''

        sampled_content = []
        qids = random.sample(set(queries.qid), n_tasks)
        sampled_queries = [queries[queries.qid == qid].reset_index(drop = True).at[0, 'query'] for qid in qids] 
        
        for query in sampled_queries:
            # positive sample
            pos_ranking = 0
            for i in range(100):
                if not queries[(queries.loc[:,"query"] == query) & (queries.ranking == i)].empty:
                    pos_ranking = i 
                    pos_sample = self.get_sample(queries, query, pos_ranking, sampled_queries, qids)
                    break
            
            # negative sample
            while True:
                neg_ranking = random.choice(list(range(pos_ranking+1, 100)))
                if not queries[(queries.loc[:,"query"] == query) & (queries.ranking == neg_ranking)].empty:
                    neg_sample = self.get_sample(queries, query, neg_ranking, sampled_queries, qids)
                    break
        
            sampled_content.append([pos_sample, neg_sample])
        assert len(sampled_queries) == len(sampled_content)
        return sampled_queries, sampled_content    

    def trans_to_index(self, texts):
        """
        将输入转化为索引表示
        :param texts: 输入格式：[], 如果is_sim为True，则格式：[[]]
        :return:
        """
        
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.__vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []

        for text in texts:
            text = tokenization.convert_to_unicode(text)
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))
        
        return input_ids, input_masks, segment_ids

    def padding(self, input_ids, input_masks, segment_ids):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids = [], [], []
        for input_id, input_mask, segment_id in zip(input_ids, input_masks, segment_ids):
            if len(input_id) < self._sequence_length:
                pad_input_ids.append(
                    input_id + [0] * (self._sequence_length - len(input_id)))
                pad_input_masks.append(
                    input_mask + [0] * (self._sequence_length - len(input_mask)))
                pad_segment_ids.append(
                    segment_id + [0] * (self._sequence_length - len(segment_id)))
            else:
                pad_input_ids.append(input_id[:self._sequence_length])
                pad_input_masks.append(input_mask[:self._sequence_length])
                pad_segment_ids.append(segment_id[:self._sequence_length])

        return pad_input_ids, pad_input_masks, pad_segment_ids

    def gen_data(self, train_file, eval_file):
        """
        生成数据
        :param file_path:
        :return:
        """

        # 1，读取原始数据
        train_queries, eval_queries = self.load_data(train_file, eval_file)
        print("read finished")

        return train_queries, eval_queries

    def gen_task_samples(self, queries, n_tasks):
        """
        生成训练任务和验证任务
        :param queries:
        :param n_tasks:
        :return:
        """
        # 1, 采样
        text_as, text_bs = self.neg_samples(queries, n_tasks)
        self.count += 1
        print("sample {}".format(self.count))

        # 2，输入转索引
        input_ids_a, input_masks_a, segment_ids_a = self.trans_to_index(
            text_as)
        input_ids_a, input_masks_a, segment_ids_a = self.padding(
            input_ids_a, input_masks_a, segment_ids_a)

        input_ids_b, input_masks_b, segment_ids_b = [], [], []
        for text_b in text_bs:
            input_id_b, input_mask_b, segment_id_b = self.trans_to_index(
                text_b)
            input_id_b, input_mask_b, segment_id_b = self.padding(
                input_id_b, input_mask_b, segment_id_b)
            input_ids_b.append(input_id_b)
            input_masks_b.append(input_mask_b)
            segment_ids_b.append(segment_id_b)

        return input_ids_a, input_masks_a, segment_ids_a, input_ids_b, input_masks_b, segment_ids_b

    def next_batch(self, input_ids_a, input_masks_a, segment_ids_a, input_ids_b, input_masks_b, segment_ids_b):
        """
        生成batch数据
        :param input_ids_a:
        :param input_masks_a:
        :param segment_ids_a:
        :param input_ids_b:
        :param input_masks_b:
        :param segment_ids_b:
        :return:
        """
        print("num of epoch: ", len(input_ids_a))
        num_batches = len(input_ids_a) // self._batch_size

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size
            batch_input_ids_a = input_ids_a[start: end]
            batch_input_masks_a = input_masks_a[start: end]
            batch_segment_ids_a = segment_ids_a[start: end]

            batch_input_ids_b = input_ids_b[start: end]
            batch_input_masks_b = input_masks_b[start: end]
            batch_segment_ids_b = segment_ids_b[start: end]

            yield dict(input_ids_a=batch_input_ids_a,
                       input_masks_a=batch_input_masks_a,
                       segment_ids_a=batch_segment_ids_a,
                       input_ids_b=list(chain(*batch_input_ids_b)),
                       input_masks_b=list(chain(*batch_input_masks_b)),
                       segment_ids_b=list(chain(*batch_segment_ids_b)))
