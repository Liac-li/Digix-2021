import json
import os
import pandas as pd



class FinalPredict(object):
	def __init__(self, configs):
		self.configs = configs
		self.__output_path = configs['output_path']
		if not os.path.exists(self.__output_path):
			os.makedirs(self.__output_path)

		self.tr_recall_path = os.path.join(configs['recall_path'], 'tr_recall_10.csv')
		self.en_recall_path = os.path.join(configs['recall_path'], 'en_recall_10.csv')

	
	def gen_predict_data(self, to_predict_path, type='a'):
		'''
		Generate data to predict like type [[query, text1, text2], []]
		@param type: output type to gen:
			a: all top100 text in single list [[query, text1, ..., text100]]
			b: all top100 text in different list with same query [[query, text1], [query, text2],...]]
		:return : a list consists of query with text list
		'''
		df_predict = pd.read_csv(to_predict_path, encoding='utf8')
		df_tr_recall = pd.read_csv(self.tr_recall_path, encoding='utf8')
		df_en_recall = pd.read_csv(self.en_recall_path, encoding='utf8')


		# gen target data one by one on query
		for index, row in df_predict.iterrows():
			pid, query = row['pid'], row['query']
			if pid[:2] == 'tr':
				df_match = df_tr_recall[df_tr_recall['pid'] == pid]
			elif pid[:2] == 'en':
				df_match = df_en_recall[df_en_recall['pid'] == pid]
			
			if len(df_match) != 100:
				print(f"[gen_predict_data] {pid} matchs only {len(df_match)} texts")

			
		


