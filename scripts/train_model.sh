source /home/getalp/sfeirj/anaconda3/bin/activate env_1

CUDA_VISIBLE_DEVICES=0


# Remove old checkpoints
rm /home/getalp/sfeirj/sfeirseq/checkpoints/*

# Train model
fairseq-train /home/getalp/sfeirj/sfeirseq/maxmentionsfr-bin \
				  -a transformer \
				  --optimizer adam \
				  --lr 0.0005 \
				  --max-update 500000 \
				  -s input -t label \
				  --batch-size 50 \
				  --encoder-layers 2 \
				  --decoder-layers 2 \
				  --encoder-embed-dim 64 \
				  --decoder-embed-dim 64 \
				  --dropout 0.1 \
				  --encoder-attention-heads 8 \
				  --decoder-attention-heads 8 \
				  --save-dir /home/getalp/sfeirj/sfeirseq/checkpoints

# Predict via --gen-subset subset and evaluate BLEU4 score
python /home/getalp/sfeirj/sfeirseq/generate.py \
				  /home/getalp/sfeirj/sfeirseq/maxmentionsfr-bin/ \
				  --path /home/getalp/sfeirj/sfeirseq/checkpoints/checkpoint_best.pt \
				  --gen-subset valid

# Transform predicted data into conll-format word by word file
# (not using dataframes immensely accelerates the process)
python /home/getalp/sfeirj/scripts/conlleval/preprocess_for_conlleval_no_df.py

# Evaluate via conll-eval
/home/getalp/sfeirj/scripts/conlleval/conlleval.pl < /home/getalp/sfeirj/data/valid_predictions_for_eval | head
