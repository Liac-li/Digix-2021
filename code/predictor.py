import os
import json
import random
import argparse
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from bert import modeling
from model import BertPairLTR
from data_helper import TrainData
from metrics import mean, accuracy


class Predictor(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)

        #self.__bert_checkpoint_path = os.path.join(self.config["bert_model_path"], "bert_model.ckpt")
        self.ckpt_path = self.config['ckpt_model_path']
        # 加载数据集
        self.data_obj = self.load_data()
        #self.queries = self.data_obj.gen_data(self.config["data"])
        # self.queries = self.data_obj.load_predict_data()

        # print("test data size: {}".format(len(self.queries)))

        tf.set_random_seed(42)

        num_train_steps = int(self.config["train_n_tasks"] /
                              self.config["batch_size"] *
                              self.config["epochs"])
        num_warmup_steps = int(num_train_steps * self.config["warmup_rate"])

        # 初始化模型对象
        self.model = BertPairLTR(config=self.config,
                                 is_training=False,
                                 num_train_step=num_train_steps,
                                 num_warmup_step=num_warmup_steps)

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        data_obj = TrainData(self.config)
        return data_obj

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        # for op in tf.get_default_graph().get_operations():
        #     print(op.name)

        self.similarity = tf.get_default_graph().get_tensor_by_name(
            "cosine_similarity/similarity:0")
        self.predictions = tf.get_default_graph().get_tensor_by_name(
            "cosine_similarity/predictions:0")

        self.input_ids_a = tf.get_default_graph().get_tensor_by_name(
            "input_ids_a:0")
        self.input_masks_a = tf.get_default_graph().get_tensor_by_name(
            "input_mask_a:0")
        self.segment_ids_a = tf.get_default_graph().get_tensor_by_name(
            "segment_ids_a:0")
        self.input_ids_b = tf.get_default_graph().get_tensor_by_name(
            "input_ids_b:0")
        self.input_masks_b = tf.get_default_graph().get_tensor_by_name(
            "input_mask_b:0")
        self.segment_ids_b = tf.get_default_graph().get_tensor_by_name(
            "segment_ids_b:0")

        feed_dict = {
            self.input_ids_a: batch["input_ids_a"],
            self.input_masks_a: batch["input_masks_a"],
            self.segment_ids_a: batch["segment_ids_a"],
            self.input_ids_b: batch["input_ids_b"],
            self.input_masks_b: batch["input_masks_b"],
            self.segment_ids_b: batch["segment_ids_b"]
        }

        # predict, similarity = sess.run([self.predictions, self.similarity], feed_dict=feed_dict)
        # similarity = sess.run(self.similarity, feed_dict=feed_dict)
        predict = sess.run(self.predict, feed_dict=feed_dict)

        return predict

    def predict(self, queries):
        print("[Predictor] test data size: {}".format(len(queries)))
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.ckpt_path +
                                                   'ltr_pair-30.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
            print("load session")
            current_step = 0

            # 测试集采样，待定
            # print(self.queries)
            t_in_ids_a, t_in_masks_a, t_seg_ids_a, t_in_ids_b, t_in_masks_b, t_seg_ids_b = \
                self.data_obj.gen_test_samples(queries)

            for batch in self.data_obj.next_test_batch(t_in_ids_a,
                                                       t_in_masks_a,
                                                       t_seg_ids_a, t_in_ids_b,
                                                       t_in_masks_b,
                                                       t_seg_ids_b):
                # predictions, similarity = self.infer(sess, batch)
                # similarity = self.model.infer(sess, batch)
                predict = self.model.infer(sess, batch)
                print("predict: step: {}".format(current_step))
                current_step += 1
                # print(similarity)
                # print(predict)
                yield predict  # 在final_submit 中作为generator调用


# python predictor.py --config_path=test_config.json
if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    Predictor = Predictor(args)
    Predictor.predict()
