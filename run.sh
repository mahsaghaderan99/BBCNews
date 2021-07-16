#!/bin/bash
labels=(
'ايران'
'هنر'
'ورزش'
'اقتصاد'
'دانش'
)

if [ $1 -eq 4 ]
then
  for l in "${labels[@]}"; do
    echo "$l"
    python src/lm/main.py --cuda --model LSTM --label $l
    python src/lm/main.py --cuda --model GRU --label $l
    python src/lm/main.py --cuda --model Transformer --label $l
  done
fi
if [ $1 -eq 41 ]
then
  for l in "${labels[@]}"; do
    echo "$l"
    python src/lm/generate.py --cuda --model LSTM --label $l
    python src/lm/generate.py --cuda --model GRU --label $l
    python src/lm/generate.py --cuda --model Transformer --label $l
  done
fi