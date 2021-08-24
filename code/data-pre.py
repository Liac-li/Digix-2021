# -*- coding: UTF-8 -*-
from bert import tokenization
import os
import sys
import copy
import json
import random
from itertools import chain
import pandas as pd
import re
from TurkishStemmer import TurkishStemmer

sys.path.append(os.path.dirname(os.getcwd()))

turkStem = TurkishStemmer()


class TrainData(object):
    def __init__(self, config):
        self.__vocab_path = os.path.join(config["bert_model_path"],
                                         "vocab.txt")
        self.__output_path = config["output_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)
        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长
        self._batch_size = config["batch_size"]

        self.__num_samples = config["num_samples"]

        self.count = 0  # sample num

        self.en_train_data_path = os.path.join(config['data_path'],
                                               'train/train_en.tsv')
        self.tr_train_data_path = os.path.join(config['data_path'],
                                               'train/train_tr.tsv')
        self.predict_path = os.path.join(config['data_path'], 'to_predict.csv')
        self.en_doc_info_path = os.path.join(config['data_path'],
                                             'doc_info/en_list_result/')
        self.tr_doc_info_path = os.path.join(config['data_path'],
                                             'doc_info/tr_list_result/')
        self.part_range = 30000  # num of file num in doc_info 'part-xxxx'

    @staticmethod
    def load_data(file_path):
        """
        :return: DataFrame for all query both in English and Turkish
        """
        data = pd.read_csv(file_path).iloc[:, 1:]
        return data

    def sentence_processing(sentence):
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
        tokens = sentence.split()
        return tokens

    def getWordsFromURL(url, lang):
        words_list = re.compile(r'[\:/?=\-&.,_@%!$0123456789()&*+\[\]]+',
                                re.UNICODE).split(url)
        drop_words = set([
            '', 'http', 'https', 'www', 'com', '\t', 'm', 'b', 'c', 'd', 'f',
            'g', 'h', 'j', 'k', 'l', 'n', 'o', 'p', 'q', 'r', 's', 't', 'v',
            'w', 'x', 'y', 'z'
        ])

        return [
            turkStem.stem(word.lower()) for word in words_list
            if word.lower() not in drop_words
        ]

    def get_tokens(self, series):
        """
            Args:
                @param series: pandas.core.series.Series with lind_index(part_num, line_num)
        """
        link_index = eval(series['link_index'])
        lang = series['qid'][:2]
        if lang == 'en':
            with open(self.en_doc_info_path + 'part-%05d' % link_index[0],
                      'r',
                      encoding='utf8') as fr:
                line = fr.readlines()[link_index[1]]
        elif lang == 'tr':
            with open(self.tr_doc_info_path + 'part-%05d' % link_index[0],
                      'r',
                      encoding='utf8') as fr:
                line = fr.readlines()[link_index[1]]
        else:
            raise NameError('Language type not match')

        parts = line.split(
            '\x01')[:3]  # ! set as url-title-rank, may need to change
        if len(parts) == 1:
            title_with_content = ''
        elif len(parts) == 2:
            title_with_content = parts[1]
        else:
            title_with_content = (parts[1] + ' ') * 20 + '.' + parts[2]
        res = self.sentence_process(
            title_with_content) + 20 * self.getWordsFromURL(parts[0], lang)
        return res

    def neg_samples(self, queries, n_tasks):
        """
        随机负采样多个样本
        Args:
            @param queries: (pd.DataFrame) all of the queries data
            @param n_tasks: set manually 

        return: [], [[]]
        """

        # adding
        # sampled_content = []
        # qids = random.sample(set(queries.qid), n_tasks)
        # sampled_queries = [queries[(queries.qid == qid) & (
        #     queries.ranking == 0)].reset_index().at[0, 'query'] for qid in qids]

        # for query in sampled_queries:
        #     # positive sample
        #     pos_sample = self.get_sample(
        #         queries, query, 0, sampled_queries, qids)
        #     # negative sample

        #     if self.__num_sample > 1:
        #     neg_sample = self.get_sample(
        #         queries, query, 10, sampled_queries, qids)

        #     sampled_content.append([pos_sample, neg_sample])
        # assert len(sampled_queries) == len(sampled_content)
        # return sampled_queries, sampled_content

        new_queries = []
        new_sims = []

        qids = random.sample(set(queries.qid), n_tasks)
        cor_qids = list(set(queries.qid) - set(qids))

        for i in range(n_tasks):
            pos_q = queries[(queries['qid'] == qids[i])
                            & (queries['ranking'] == 0)]
            neg_sim = random.sample(cor_qids, self.__num_samples - 1)
            neg_q = queries[queries['qid'].isin(neg_sim)].sample(
                n=self.__num_samples - 1)

            tmp = []
            for index, item in pos_q.append(neg_q).iterrows():
                tmp.append(get_tokens(item))

            new_queries.append(pos_q['query'])
            new_sims.append(tmp)
        return new_queries, new_sims

    def trans_to_index(self, texts):
        """
        将输入转化为索引表示
        :param texts: 输入格式：[], 如果is_sim为True，则格式：[[]]
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path,
                                               do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []

        for text in texts:
            if isinstance(text, list):
                tokens = text
            else:
                text = tokenization.convert_to_unicode(text)
                tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            print(tokens)
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
        for input_id, input_mask, segment_id in zip(input_ids, input_masks,
                                                    segment_ids):
            if len(input_id) < self._sequence_length:
                pad_input_ids.append(input_id + [0] *
                                     (self._sequence_length - len(input_id)))
                pad_input_masks.append(
                    input_mask + [0] *
                    (self._sequence_length - len(input_mask)))
                pad_segment_ids.append(
                    segment_id + [0] *
                    (self._sequence_length - len(segment_id)))
            else:
                pad_input_ids.append(input_id[:self._sequence_length])
                pad_input_masks.append(input_mask[:self._sequence_length])
                pad_segment_ids.append(segment_id[:self._sequence_length])

        return pad_input_ids, pad_input_masks, pad_segment_ids

    def gen_data(self, file_path):
        """
        生成数据
        :param file_path:
        :return:
        """

        # 1，读取原始数据
        queries = self.load_data(file_path)
        print("read finished")

        return queries

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

    def next_batch(self, input_ids_a, input_masks_a, segment_ids_a,
                   input_ids_b, input_masks_b, segment_ids_b):
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
            batch_input_ids_a = input_ids_a[start:end]
            batch_input_masks_a = input_masks_a[start:end]
            batch_segment_ids_a = segment_ids_a[start:end]

            batch_input_ids_b = input_ids_b[start:end]
            batch_input_masks_b = input_masks_b[start:end]
            batch_segment_ids_b = segment_ids_b[start:end]

            yield dict(input_ids_a=batch_input_ids_a,
                       input_masks_a=batch_input_masks_a,
                       segment_ids_a=batch_segment_ids_a,
                       input_ids_b=list(chain(*batch_input_ids_b)),
                       input_masks_b=list(chain(*batch_input_masks_b)),
                       segment_ids_b=list(chain(*batch_segment_ids_b)))
