import numpy as np
import argparse
import json
import os

# from data_helper import TrainData
from predictor import Predictor
from data_helper import TrainData


class Submiting(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)
        self.__output_path = self.config['output_path']
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.data_obj = self.load_data()
        self.predict_model = self.load_predict_model()

        self._batch_size = self.config['batch_size']

        # if not os.path.isfile(self.config['output_path']+'test_data.json'):
        #     self.data_obj.gen_predict_data(self.config['predict_file'])

    def load_predict_model(self):
        predict_model = Predictor(self.args)
        return predict_model

    def load_data(self):
        data_obj = TrainData(self.config)
        if not os.path.isfile(self.__output_path + 'test_data.json'):
            data_obj.gen_predict_data()
        return data_obj

    @staticmethod
    def gen_top10_urls(buffer):
        """
        @param buffer: [(score, url)]
        """
        buffer = sorted(buffer, key=lambda item: item[0], reverse=True)
        buffer = buffer[:10]
        buffer = [item[-1] for item in buffer]
        return '\x01'.join(buffer)

    def get_origin_file(self, output_path):
        queries = self.data_obj.load_predict_data()
        cnt = 0
        list_buffer = []
        last_query = None

        if not os.path.isfile(output_path):
            with open(output_path, 'w', encoding='utf8') as fw:
                fw.write('qid\x01doc\n')

        for prediction in self.predict_model.predict(queries):
            text = None
            start = cnt * self._batch_size
            end = start + self._batch_size

            urls = [item[-1].split('\x01')[-1] for item in queries[start:end]]
            query_list = [item[0] for item in queries[start:end]]
            for idx in range(len(query_list)):
                if last_query is None:
                    last_query = query_list[idx]
                elif last_query != query_list[idx]:
                    list_buffer += list(zip(prediction[:idx], urls[idx]))
                    text = self.gen_top10_urls(list_buffer)
                    text = queries[start][-1].split('\x01')[0] + '\x01' + text
                    # reset status
                    last_query = query_list[idx]
                    list_buffer = []
                    urls = urls[idx:]
                    prediction = prediction[idx:]

            list_buffer += list(zip(prediction, urls))
            if text is not None:
                with open(output_path, 'a', encoding='utf8') as fa:
                    fa.write(text + '\n')

            print("prediction is ", prediction)
            cnt += 1


# python final_submit.py --config_path=test_config.json
if __name__ == "__main__":
    # Read user's config
    paraser = argparse.ArgumentParser()
    paraser.add_argument("--config_path", help="config path of model")
    args = paraser.parse_args()
    Submiting = Submiting(args)

    Submiting.get_origin_file('/tmp/foo/test')
