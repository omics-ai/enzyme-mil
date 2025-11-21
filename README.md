# Enzyme EC number prediction with MIL

## dataset

* training set: ``dataset/split100.csv``
* test set 1: ``dataset/new.csv``
* test set 2: ``dataset/price.csv``
* test set 3: ``dataset/uniprot-all.csv``
* test set 4: ``dataset/uniprot-multi.csv``

## prepare domain instances

[DCTdomain](https://github.com/mgtools/DCTdomain) provided protein domain-level embeddings. 

Here is an example.
```python
git clone https://github.com/mgtools/DCTdomain.git
cd DCTdomain
python src/make_db.py --fafile split100.fasta --dbfile output/split100.db --gpu 1 --cpu 16
```

## model training

Our models were trained on a single Nvidia H100 in Python 3.12 with PyTorch 2.9.

```python
cd code
python mil-bc_emb.py --config_path config_milbc.yaml
```

----

## pretrained models

[CLEAN](https://github.com/tttianhao/CLEAN) shared pretrained model on [Google Drive](https://drive.google.com/file/d/1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ/view?usp=sharing)

[EnzHier](https://github.com/labxscut/EnzHier) shared pretrained model on [GitHub](https://github.com/TangTZscut/EnzHier/blob/main/data/model/split100_triplet_withEC_7000.pth)

