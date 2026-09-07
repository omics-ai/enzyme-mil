#!/usr/bin/env python
"""Turn a table of protein sequences into the feature archive the three models read.

    python process.py --csvdata ../example/example.csv

Writes ``../example/example.npz`` -- the input CSV's path with a ``.npz`` suffix, unless
``--output`` says otherwise. No EC labels are needed or read: a column of them may be present and
is ignored, so the same command works on completely unannotated proteins.

Two feature sets go into the archive.

**DCTdomain domain embeddings.** DCTdomain segments each protein into domains and emits one
480-value DCT fingerprint per domain plus one for the whole protein, quantised to ``int8`` in
``[0, 127]``. Dividing by 128 puts them in ``[0, 1)``. Up to 16 domains are kept, padded with zeros
and tracked by a boolean mask; a protein DCTdomain finds no domain in falls back to a single token
holding its whole-protein fingerprint. ``dct_feature`` is stored 960 wide as
``[local | whole-protein repeated]``, which is the layout the benchmark archives use.

**ESM2 whole-sequence embedding.** The mean of the layer-33 residue representations of
``esm2_t33_650M_UR50D``, excluding the BOS and EOS tokens, over the full untruncated sequence.

DCTdomain ships in ``enzyme-mil/DCTdomain`` and runs as a subprocess, in this same interpreter
unless ``--dct-python`` says otherwise. It needs its ``RecCut`` domain-cutter compiled once:

    cd DCTdomain && g++ -o src/RecCut src/RecCut.cpp

``--dctdomain`` (or ``DCTDOMAIN_HOME``) points at a different clone. Already have fingerprints?
``--dct-npz`` skips that step entirely.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

MAX_DOMAINS = 16
DOMAIN_DIM = 480
ESM_DIM = 1280
ESM_LAYER = 33
ESM_MODEL = "esm2_t33_650M_UR50D"
QUANTISATION = 128.0
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZUO")


def find_dctdomain(explicit: str | None) -> Path:
    """Locate DCTdomain: the flag, then ``DCTDOMAIN_HOME``, then the copy bundled beside ``code/``.

    Also checks that ``RecCut`` has been compiled. It is a C++ domain cutter that DCTdomain calls
    per sequence, and no binary ships with this package, so the failure is otherwise a confusing
    one from deep inside DCTdomain rather than a missing build step.
    """
    here = Path(__file__).resolve().parent
    candidates = [explicit, os.environ.get("DCTDOMAIN_HOME"), here.parent / "DCTdomain"]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        if not (root / "src" / "make_db.py").is_file():
            continue
        cutter = root / "src" / "RecCut"
        if not (cutter.is_file() and os.access(cutter, os.X_OK)):
            raise SystemExit(
                f"DCTdomain at {root} has no compiled RecCut. Build it once:\n"
                f"    cd {root} && g++ -o src/RecCut src/RecCut.cpp"
            )
        return root
    raise SystemExit(
        f"cannot find DCTdomain. Expected it at {here.parent / 'DCTdomain'}; pass --dctdomain "
        f"/path/to/DCTdomain or export DCTDOMAIN_HOME to use a clone from "
        f"https://github.com/mgtools/DCTdomain instead."
    )


def read_sequences(path: Path) -> list[tuple[str, str]]:
    """``(Entry, Sequence)`` pairs from a tab- or comma-separated table, in file order."""
    text = path.read_text()
    if not text.strip():
        raise SystemExit(f"{path} is empty")
    delimiter = "\t" if text.splitlines()[0].count("\t") >= text.splitlines()[0].count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    if not rows:
        raise SystemExit(f"{path} has a header but no rows")
    for column in ("Entry", "Sequence"):
        if column not in rows[0]:
            raise SystemExit(f"{path} needs an '{column}' column; found {sorted(k for k in rows[0] if k)}")

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=2):
        entry = (row["Entry"] or "").strip()
        sequence = (row["Sequence"] or "").strip().upper()
        if not entry:
            raise SystemExit(f"{path} line {number}: empty Entry")
        if entry in seen:
            raise SystemExit(f"{path} line {number}: duplicate Entry {entry}")
        if not sequence:
            raise SystemExit(f"{path} line {number}: {entry} has no sequence")
        unknown = set(sequence) - AMINO_ACIDS
        if unknown:
            raise SystemExit(f"{path} line {number}: {entry} has non-amino-acid characters {sorted(unknown)}")
        seen.add(entry)
        pairs.append((entry, sequence))
    return pairs


def run_dctdomain(
    pairs: list[tuple[str, str]],
    workdir: Path,
    dctdomain: Path,
    interpreter: str,
    gpus: int,
    cpus: int,
    maxlen: int,
) -> Path:
    """Fingerprint every sequence and return the ``*.db-dct.npz`` DCTdomain writes."""
    fasta = workdir / "sequences.fasta"
    with fasta.open("w") as handle:
        for entry, sequence in pairs:
            handle.write(f">{entry}\n{sequence}\n")
    database = workdir / "sequences.db"
    command = [
        interpreter,
        str(dctdomain / "src" / "make_db.py"),
        "--fafile",
        str(fasta),
        "--dbfile",
        str(database),
        "--maxlen",
        str(maxlen),
        "--cpu",
        str(cpus),
        # Nothing here reads the FAISS similarity index, so skip building it.
        "--noindex",
    ]
    if gpus > 0:
        command += ["--gpu", str(gpus)]
    print(f"[1/2] DCTdomain fingerprints for {len(pairs)} sequences", flush=True)
    # Run from the work directory: DCTdomain writes per-sequence scratch files into the working
    # directory, and Python resolves its imports from the script's own directory regardless.
    result = subprocess.run(command, cwd=workdir)
    if result.returncode != 0:
        raise SystemExit(
            f"DCTdomain failed with exit code {result.returncode}. It needs torch, fair-esm, scipy "
            f"and faiss in the interpreter running it ({interpreter}); see requirements.txt, or "
            f"pass --dct-python to use a different one."
        )
    fingerprints = database.with_name(database.name + "-dct.npz")
    if not fingerprints.is_file():
        raise SystemExit(f"DCTdomain produced no fingerprint archive at {fingerprints}")
    return fingerprints


def read_fingerprints(path: Path) -> dict[str, np.ndarray]:
    """``entry -> (n_domains + 1, 480)`` DCT fingerprints; the last row is the whole protein."""
    with np.load(path, allow_pickle=True) as archive:
        missing = {"sid", "idx", "dct"} - set(archive.files)
        if missing:
            raise SystemExit(f"{path} is missing {sorted(missing)}; is it a DCTdomain *-dct.npz?")
        identifiers, offsets, fingerprints = archive["sid"], archive["idx"], archive["dct"]
    if len(offsets) != len(identifiers) + 1:
        raise SystemExit(f"{path}: idx must hold one offset per sequence plus a final bound")
    if fingerprints.ndim != 2 or fingerprints.shape[1] != DOMAIN_DIM:
        raise SystemExit(f"{path}: expected {DOMAIN_DIM}-value fingerprints, found {fingerprints.shape}")
    return {
        str(identifier): fingerprints[offsets[position] : offsets[position + 1], :]
        for position, identifier in enumerate(identifiers)
    }


def dct_to_representation(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad the DCT fingerprints of one protein into the ``(16, 960)`` bag the models read.

    ``feature`` is what DCTdomain emitted for this protein: one row per domain followed by one
    whole-protein row. Every valid token is ``[its own domain | the whole protein]``, and a protein
    with no domain keeps a single token where both halves are the whole-protein fingerprint.
    """
    dct_count = feature.shape[0] - 1
    representation = np.full((MAX_DOMAINS, 2 * DOMAIN_DIM), 0, dtype=float)
    mask = np.full((MAX_DOMAINS,), False, dtype=bool)
    if dct_count > MAX_DOMAINS:
        print(f"[warn] {dct_count} domains found; keeping the first {MAX_DOMAINS}", flush=True)
    if dct_count > 0:
        for position in range(min(dct_count, MAX_DOMAINS)):
            representation[position, :DOMAIN_DIM] = feature[position, :] / QUANTISATION
            representation[position, DOMAIN_DIM:] = feature[-1, :] / QUANTISATION
            mask[position] = True
    else:
        representation[0, :DOMAIN_DIM] = feature[0, :] / QUANTISATION
        representation[0, DOMAIN_DIM:] = feature[-1, :] / QUANTISATION
        mask[0] = True
    return representation, mask, dct_count


@torch.inference_mode()
def embed_sequences(pairs: list[tuple[str, str]], device: str, batch_tokens: int) -> dict[str, np.ndarray]:
    """Mean-pooled ESM2 layer-33 embeddings, batched shortest-first to limit padding waste."""
    try:
        import esm
    except ImportError as error:  # pragma: no cover - environment problem, not a code path
        raise SystemExit("fair-esm is required for the ESM2 embedding: pip install fair-esm") from error

    print(f"[2/2] ESM2 ({ESM_MODEL}) embeddings on {device}", flush=True)
    model, alphabet = getattr(esm.pretrained, ESM_MODEL)()
    model = model.eval().to(device)
    converter = alphabet.get_batch_converter()

    order = sorted(range(len(pairs)), key=lambda position: len(pairs[position][1]))
    embeddings: dict[str, np.ndarray] = {}
    batch: list[int] = []
    done = 0

    def flush(indices: list[int]) -> None:
        nonlocal done
        if not indices:
            return
        _, _, tokens = converter([pairs[position] for position in indices])
        representations = model(tokens.to(device), repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
        for row, position in enumerate(indices):
            entry, sequence = pairs[position]
            # Skip the BOS token and stop before EOS and any padding: mean over residues alone.
            vector = representations[row, 1 : len(sequence) + 1].mean(dim=0)
            embeddings[entry] = vector.float().cpu().numpy().astype(np.float32)
        done += len(indices)
        print(f"      {done}/{len(pairs)} sequences", flush=True)

    for position in order:
        longest = max(len(pairs[index][1]) for index in [*batch, position]) + 2
        if batch and longest * (len(batch) + 1) > batch_tokens:
            flush(batch)
            batch = []
        batch.append(position)
    flush(batch)
    return embeddings


def build_samples(
    pairs: list[tuple[str, str]],
    fingerprints: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    """One sample dict per protein, in input order."""
    samples = []
    for entry, _ in pairs:
        if entry not in fingerprints:
            raise SystemExit(f"DCTdomain returned no fingerprint for {entry}")
        representation, mask, dct_count = dct_to_representation(fingerprints[entry])
        embedding = embeddings[entry]
        if embedding.shape != (ESM_DIM,):
            raise SystemExit(f"{entry}: ESM2 embedding is {embedding.shape}, expected ({ESM_DIM},)")
        if not (np.isfinite(representation).all() and np.isfinite(embedding).all()):
            raise SystemExit(f"{entry}: non-finite feature")
        samples.append(
            {
                "entry": entry,
                "dct_count": dct_count,
                "dct_feature": representation,
                "dct_global": representation[0, DOMAIN_DIM:].copy(),
                "dct_mask": mask,
                "esm2_feature": embedding,
            }
        )
    archive = np.empty(len(samples), dtype=object)
    archive[:] = samples
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csvdata", required=True, help="table with 'Entry' and 'Sequence' columns")
    parser.add_argument("--output", help="output .npz (default: --csvdata with a .npz suffix)")
    parser.add_argument("--dctdomain", help="DCTdomain clone (default: the bundled ../DCTdomain)")
    parser.add_argument(
        "--dct-python",
        default=os.environ.get("DCTDOMAIN_PYTHON", sys.executable),
        help="interpreter to run DCTdomain with (default: $DCTDOMAIN_PYTHON, else this one)",
    )
    parser.add_argument("--dct-npz", help="reuse an existing DCTdomain *-dct.npz instead of running it")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-tokens", type=int, default=16384, help="ESM2 tokens per batch, padding included")
    parser.add_argument("--dct-gpus", type=int, default=1, help="GPUs for DCTdomain embedding; 0 for CPU")
    parser.add_argument("--dct-cpus", type=int, default=4, help="CPU workers for DCT fingerprinting")
    parser.add_argument("--dct-maxlen", type=int, default=500, help="DCTdomain sequence-splitting threshold")
    parser.add_argument("--keep-intermediate", action="store_true", help="keep the FASTA and DCTdomain database")
    args = parser.parse_args()

    source = Path(args.csvdata).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"no such file: {source}")
    destination = Path(args.output).expanduser().resolve() if args.output else source.with_suffix(".npz")
    if args.batch_tokens < 1:
        raise SystemExit("--batch-tokens must be at least one")

    pairs = read_sequences(source)
    lengths = [len(sequence) for _, sequence in pairs]
    print(f"{len(pairs)} proteins from {source}, {min(lengths)}-{max(lengths)} residues")

    if args.dct_npz:
        fingerprint_path = Path(args.dct_npz).expanduser().resolve()
        if not fingerprint_path.is_file():
            raise SystemExit(f"no such file: {fingerprint_path}")
        print(f"[1/2] reusing DCTdomain fingerprints from {fingerprint_path}")
        workdir = None
    else:
        dctdomain = find_dctdomain(args.dctdomain)
        if args.keep_intermediate:
            workdir = destination.parent / f"{destination.stem}_dct"
        else:
            workdir = Path(tempfile.mkdtemp(prefix="enzyme-mil-dct-"))
        workdir.mkdir(parents=True, exist_ok=True)
        fingerprint_path = run_dctdomain(
            pairs, workdir, dctdomain, args.dct_python, args.dct_gpus, args.dct_cpus, args.dct_maxlen
        )

    try:
        fingerprints = read_fingerprints(fingerprint_path)
        embeddings = embed_sequences(pairs, args.device, args.batch_tokens)
        samples = build_samples(pairs, fingerprints, embeddings)
    finally:
        if workdir is not None and not args.keep_intermediate:
            shutil.rmtree(workdir, ignore_errors=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, dataset_samples=samples)
    domains = [int(sample["dct_mask"].sum()) for sample in samples]
    print(
        f"\nwrote {destination}\n"
        f"  {len(samples)} proteins, {sum(domains) / len(domains):.2f} domain tokens each "
        f"(min {min(domains)}, max {max(domains)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
