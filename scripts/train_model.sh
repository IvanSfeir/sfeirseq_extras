source /home/getalp/sfeirj/anaconda3/bin/activate env_1

CUDA_VISIBLE_DEVICES=0


# Remove old checkpoints
rm /home/getalp/sfeirj/sfeirseq/checkpoints/*
rm /home/getalp/sfeirj/data/valid_predictions
rm /home/getalp/sfeirj/data/valid_predictions_for_eval

# Train model
fairseq-train /home/getalp/sfeirj/sfeirseq/maxmentions-bin \
				  -a transformer \
				  --optimizer adam \
				  --lr 0.0005 \
				  --max-update 500000 \
				  -s input -t label \
				  --batch-size 50 \
				  --encoder-layers 1 \
				  --decoder-layers 1 \
				  --encoder-embed-dim 128 \
				  --decoder-embed-dim 128 \
				  --dropout 0.2 \
				  --encoder-attention-heads 8 \
				  --decoder-attention-heads 8 \
				  --save-dir /home/getalp/sfeirj/sfeirseq/checkpoints

# Predict via --gen-subset subset and evaluate BLEU4 score
python /home/getalp/sfeirj/sfeirseq/generate.py \
				  /home/getalp/sfeirj/sfeirseq/maxmentions-bin/ \
				  --path /home/getalp/sfeirj/sfeirseq/checkpoints/checkpoint_best.pt \
				  --gen-subset valid \
				  --beam 1

# Transform predicted data into conll-format word by word file and evaluate via conll-eval
source /home/getalp/sfeirj/scripts/conlleval/preprocess_and_conlleval.sh