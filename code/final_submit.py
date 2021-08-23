import numpy as np
import argparse
import json
import os

# from data_helper import TrainData
from predictor import Predictor

class Submiting(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)

        self.predict_model = self.load_predict_model()

        self.__output_path = self.config['output_path']
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)
        

    def load_predict_model(self):
        predict_model = Predictor(self.args)
        return predict_model
    
    def get_origin_file(self):
        self.predict_model.predict()
    
    
        

# python final_submit.py --config_path=test_config.json
if __name__ == "__main__":
    # Read user's config
    paraser = argparse.ArgumentParser()
    paraser.add_argument("--config_path", help="config path of model")
    args = paraser.parse_args()
    Submiting = Submiting(args)
    Submiting.get_origin_file()

    
