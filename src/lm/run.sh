#!/bin/bash
if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en --dev-src=./chr_en_data/dev.chr --dev-tgt=./chr_en_data/dev.en --vocab=vocab.json --cuda --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "test" ]; then
   CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./chr_en_data/test.chr ./chr_en_data/test.en outputs/test_outputs.txt --cuda
fi
