# Hybrid4SentenceARA
This page released the datasets and codes in our BEA 2023 Workshop paper *Hybrid Models for Sentence Readability Assessment*.

## Datasets
**Wall Street Journal (WSJ)**

- Please refer to [Brunato et al. (2018)](https://aclanthology.org/D18-1289.pdf) and their [website](http://www.italianlp.it/resources/). 

**OneStopEnglish (OSE)**
- Please refer to [Vajjala and Luˇci ́c (2018)](https://aclanthology.org/W18-0535.pdf).
- The directory *Data* contains our processed OSE dataset.

## Codes
Train models
```python
python code/main.py \
    --model AutoModel \
    --state classification \
    --batch_size 32 \
    --epoch_num 10 \
    --embedding_dim 64 \
    --lr 1e-5 \
    --n_labels 3 \
    --dataset ./datasets/ose/ 
```

Obtain predicted value or probabilities
```python
python code/pred.py \
    ./trained_models/ose/roberta_32_1e-06_10/ \
    ./datasets/traindata/ose/ \
    ./Pred_results/ose/
```
