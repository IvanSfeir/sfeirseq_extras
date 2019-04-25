source /home/getalp/sfeirj/anaconda3/bin/activate env_1

CUDA_VISIBLE_DEVICES=0

rm /home/getalp/sfeirj/sfeirseq/checkpoints/*

fairseq-train /home/getalp/sfeirj/sfeirseq/maxmentions-bin \
				  -a transformer \
				  --optimizer adam \
				  --lr 0.0005 \
				  --max-update 50000 \
				  -s input -t label \
				  --batch-size 50 \
				  --encoder-layers 2 \
				  --decoder-layers 2 \
				  --encoder-embed-dim 64 \
				  --decoder-embed-dim 64 \
				  --dropout 0.01 \
				  --encoder-attention-heads 8 \
				  --decoder-attention-heads 8

python /home/getalp/sfeirj/sfeirseq/generate.py \
				  /home/getalp/sfeirj/sfeirseq/maxmentions-bin/ \
				  --path /home/getalp/sfeirj/sfeirseq/checkpoints/checkpoint_best.pt \
				  --gen-subset valid \
				  --remove-bpe