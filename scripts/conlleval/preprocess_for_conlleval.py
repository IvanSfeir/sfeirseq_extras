import pandas as pd
from tqdm import tqdm

parent_path = "/home/getalp/sfeirj/data/"

# Build dataframe
predictions = pd.read_csv(parent_path+"valid_predictions", delimiter="\\t", \
	names=["source","target","prediction"])


def preprocess():
	# Saves predictions in a CoNLL-friendly format in parent path
	# Output data is ready for conlleval.pl script

	conll_list = []

	for row_idx,row in tqdm(predictions.iterrows()):
		splitted_source = row["source"].split(" ")
		splitted_target = row["target"].split(" ")
		splitted_prediction = row["prediction"].split(" ")
	    
	    # If prediction and target have same length
		if len(splitted_target) == len(splitted_prediction):
			for idx in range(len(splitted_target)):
				conll_list.append([row_idx, splitted_source[idx], \
	    									splitted_target[idx], splitted_prediction[idx]])
	            
	    # If target is longer
		elif len(splitted_target) > len(splitted_prediction):
			for idx in range(len(splitted_prediction)):
				conll_list.append([row_idx, splitted_source[idx], \
											splitted_target[idx], splitted_prediction[idx]])
	        # Fill prediction with "O" tags
			for idx in range(len(splitted_prediction), len(splitted_target)):
				conll_list.append([row_idx, splitted_source[idx], \
											splitted_target[idx], "O"])
	            
	    # If prediction is longer
		else:
			for idx in range(len(splitted_target)):
				conll_list.append([row_idx, splitted_source[idx], \
											splitted_target[idx], splitted_prediction[idx]])
	    
	    # Write empty line after every sentence
		conll_list.append(["", "", "", ""])

	with open(parent_path+"valid_predictions_for_eval", "w") as f:
		for word_row in conll_list:
			f.write(str(word_row[0]))
			for e in word_row[1:]:
				f.write(" {}".format(str(e)))
			f.write("\n")


def get_length_differences():
	# Creates dataframe containing the sentences whose target and prediction 
	# don't have the same length
	# Returns dataframe containing exclusively wrongly predicted sizes sentences

	df_diff = pd.DataFrame(columns=["sentence_idx","source","target","prediction"])
	cter = 0
	for idx,row in predictions.iterrows():
		src_len = len(row["source"].split(" "))
		pred_len = len(row["prediction"].split(" "))
		if (src_len != pred_len):
			df_diff.loc[cter] = row
			print((src_len, pred_len))
			cter += 1
	print("{} predicted sentences among {} are of different length than target".format(cter, \
		len(predictions)))
	return df_diff


if __name__ == '__main__':
	preprocess()