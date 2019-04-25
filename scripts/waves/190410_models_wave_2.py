# Script to be launched from sfeirseq folder

import os
from itertools import product

layers_list = [2, 3, 4]
embed_dim_list = [32, 64, 128]
dropout_list = [0.05, 0.1]
lr_list = [0.0005]

gpu_number = 0

filepath = "/home/getalp/sfeirj/results/waves/results_wave_2"
columns = ['lr', 'nb_layers', 'embed_dim', 'dropout', 'epoch', 'BLEU', 'score1', 'score2', 'score3', 'score4', \
               'brevity', 'ratio', 'syslen', 'reflen', 'my_diff_count', 'my_mean', 'my_sqrt']

# Prepare file
with open(filepath, "a") as f:
	f.write(columns[0])
	for column in columns[1:]:
		f.write(",{}".format(column))
	f.write("\n")

for lr, layers, embed_dim, dropout in product(lr_list, layers_list, embed_dim_list, dropout_list):

	for file in os.listdir("checkpoints"):
		os.remove("checkpoints/{}".format(file))

	# Write input parameters
	with open(filepath, "a") as f:
		f.write("{},{},{},{},".format(lr, layers, embed_dim, dropout))

	# TRAIN
	# Write number of epochs
	os.system("CUDA_VISIBLE_DEVICES={} fairseq-train maxmentions-bin \
				  -a transformer \
				  --optimizer adam \
				  --lr {} \
				  --max-update 50000 \
				  -s input -t label \
				  --batch-size 50 \
				  --encoder-layers {} \
				  --decoder-layers {} \
				  --encoder-embed-dim {} \
				  --decoder-embed-dim {} \
				  --dropout {}".format(gpu_number, lr, layers, layers, embed_dim, embed_dim, dropout))

	# TEST
	# Write score, BP and length similarity
	os.system("CUDA_VISIBLE_DEVICES={} python generate.py maxmentions-bin/ \
				  --path checkpoints/checkpoint_best.pt".format(gpu_number))