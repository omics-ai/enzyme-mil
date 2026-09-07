# Enzyme EC number prediction with MIL

Predict EC numbers for enzyme sequences with multiple-instance learning over [DCTdomain](https://github.com/mgtools/DCTdomain) domain embeddings and ESM2.

Each enzyme is a _bag_ and its structural domains are _instances_. A gated-attention pooler learns one attention distribution per enzyme and produces independent scores over a fixed vocabulary of
**5,242 EC numbers** learned from split100. An enzyme whose true EC is outside that vocabulary can never be called correctly.

```text
example.csv  (Entry, EC number, Sequence)
     |
     |  process.py   DCTdomain -> domain vectors + mask
     |               ESM2      -> whole-sequence embedding (1280-d)
     v
example.npz
     |
     |  infer.py     one of three models
     v
outputs/example_<model>.csv   (Entry, predicted_ec, n_predicted [, predicted_score])
```

## Dataset

All the datasets are available at [Zenodo](https://zenodo.org/records/22598911).

- training set: `dataset/split100.csv`
- four test sets
  - test set 1: `dataset/new.csv`
  - test set 2: `dataset/price.csv`
  - test set 3: `dataset/uniprot-all.csv`
  - test set 4: `dataset/uniprot-multi.csv`
- sequence-simlirity constrained test sets
  - test set 5: `dataset/test70.csv`
  - test set 6: `dataset/test50.csv`

| dataset       | enzymes | EC per enzyme | EC classes present (of 5,242) |
| :------------ | ------: | ------------: | ----------------------------: |
| NEW           |     392 |          1.28 |                           177 |
| Price         |     149 |          1.02 |                            56 |
| UniProt-all   |     987 |          1.09 |                           488 |
| UniProt-multi |      77 |          2.16 |                           110 |
| Test70        |   1,057 |          1.11 |                           492 |
| Test50        |     732 |          1.06 |                           339 |

## Models

| `--model`    | Checkpoint                            | Inputs                   | Decision rule            |
| ------------ | ------------------------------------- | ------------------------ | ------------------------ |
| `milbc`      | `checkpoints/milbc/model_v1.pth`      | domain embeddings only   | sigmoid ≥ 0.5            |
| `milbc_plus` | `checkpoints/milbc_plus/model_v1.pth` | domain embeddings + ESM2 | sigmoid ≥ 0.5            |
| `milcl_plus` | `checkpoints/milcl_plus/model_v2.pt`  | domain embeddings + ESM2 | CLEAN maximum separation |

`milcl_plus` additionally needs `checkpoints/milcl_plus/centroids.npy`, the split100 EC centroids its projection is compared against;
`infer.py` picks that up automatically from beside the checkpoint.

## Install

```bash
pip install -r requirements.txt
```

DCTdomain is bundled in `DCTdomain/`. Compile its `RecCut` domain cutter once and `process.py` refuses to start without it:

```bash
cd DCTdomain
g++ -o src/RecCut src/RecCut.cpp
```

That is the whole setup. `process.py` runs DCTdomain in this same interpreter, so `requirements.txt`
covers both: `torch`, `numpy` and `fair-esm` for the models, plus `scipy` and `faiss` for DCTdomain.
(`faiss` only builds a similarity index nothing here reads, but DCTdomain imports it
unconditionally. Conda users can `conda install -c conda-forge faiss` instead of `faiss-cpu`.)

The first run downloads the ESM2 `esm2_t33_650M_UR50D` weights (2.6 GB) into the torch hub cache.

## Use

**1. Build features.** Input is a tab-separated table with `Entry` and `Sequence`
columns. An `EC number` column may be present and is ignored, so this works on unannotated proteins.

```bash
cd code
python process.py --csvdata ../example/example.csv
```

Writes `../example/example.npz`. Useful flags: `--output` and `--device`.

**2. Predict.**

```bash
python infer.py --npzdata ../example/example.npz \
                --model milbc_plus \
                --checkpoint ../checkpoints/milbc_plus/model_v1.pth
```

Writes `../outputs/example_milbc_plus.csv` and prints a summary:

```json
{
  "model": "milbc_plus",
  "rule": "sigmoid 0.5",
  "checkpoint": "../checkpoints/milbc_plus/model_v1.pth",
  "enzymes": 10,
  "classes": 5242,
  "output": "../outputs/example_milbc_plus.csv"
}
```

```csv
Entry,predicted_ec,n_predicted
A0A0H2Z7X0,2.7.7.65,1
A0A1B3PEJ0,2.7.8.7,1
```

**3. Confidence scores (optional).** `--confscore` adds a `predicted_score` column with one score
per called EC in the same order as `predicted_ec`:

```bash
python infer.py --npzdata ../example/example.npz \
                --model milbc_plus \
                --checkpoint ../checkpoints/milbc_plus/model_v1.pth \
                --confscore
```

## Files

```text
enzyme-mil/
├── requirements.txt
├── code/
│   ├── models.py                architectures for all three checkpoints
│   ├── process.py               sequences -> feature archive
│   └── infer.py                 feature archive -> EC calls
├── checkpoints/
│   ├── classes.npy              the 5,242 EC numbers, in model output order
│   ├── milbc/model_v1.pth
│   ├── milbc_plus/model_v1.pth
│   └── milcl_plus/model_v2.pt + centroids.npy
├── DCTdomain/                   cloned from https://github.com/mgtools/DCTdomain;
│                                compile src/RecCut before first use
├── example/
│   ├── example.csv              10 enzymes with sequences and true EC numbers
│   └── example.npz              features file, ready for infer.py
└── outputs/                     where infer.py writes
```

## Citation

- **DCTdomain** — Iovino, Tang and Ye, _Genome Research_, 2024. <https://github.com/mgtools/DCTdomain>
- **ESM2** — Lin et al., _Science_, 2023. <https://github.com/facebookresearch/ESM>
- **CLEAN** max-separation — Yu et al., _Science_, 2023. <https://github.com/tttianhao/CLEAN>

---

## pretrained models

[CLEAN](https://github.com/tttianhao/CLEAN) shared pretrained model on [Google Drive](https://drive.google.com/file/d/1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ/view?usp=sharing)

[EnzHier](https://github.com/labxscut/EnzHier) shared pretrained model on [GitHub](https://github.com/TangTZscut/EnzHier/blob/main/data/model/split100_triplet_withEC_7000.pth)
