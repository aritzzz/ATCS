# ATCS

For now, only MNLI is implemented.
To train a baseline model on the MNLI dataset run:

`python train.py`

To train on the stance dataset, run:

`python train.py --n\_classes 1 --dataset\_name stance --loss bce`

For the paraphrase dataset run:

`python train.py --n\_classes 1 --dataset\_name paraphrase --loss bce`
