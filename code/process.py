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

``--dctdomain`` (or ``DCTDOMAIN_HOME``) points at a different clone; note that a stock upstream
clone lacks the bundled copy's fix for sequences dropped from the final batch. Already have
fingerprints? ``--dct-npz`` skips that step entirely.

``--device`` places the ESM2 embedding only. DCTdomain has its own ``--dct-gpus``, which always
takes CUDA devices 0 to n-1 and defaults to one GPU when CUDA is available and to CPU otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import shutil
import sqlite3
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
# DCTdomain silently skips anything shorter: it cannot fill a 3 x 80 fingerprint from it.
MIN_LENGTH = 5


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
        # DCTdomain keeps only the first word of a FASTA header as the identifier and names a
        # scratch file after it, so either of these loses the protein partway through the run.
        if "/" in entry or any(character.isspace() for character in entry):
            raise SystemExit(f"{path} line {number}: Entry {entry!r} must not contain whitespace or '/'")
        if not sequence:
            raise SystemExit(f"{path} line {number}: {entry} has no sequence")
        unknown = set(sequence) - AMINO_ACIDS
        if unknown:
            raise SystemExit(f"{path} line {number}: {entry} has non-amino-acid characters {sorted(unknown)}")
        if len(sequence) < MIN_LENGTH:
            raise SystemExit(
                f"{path} line {number}: {entry} has {len(sequence)} residues; "
                f"DCTdomain cannot fingerprint fewer than {MIN_LENGTH}"
            )
        seen.add(entry)
        pairs.append((entry, sequence))
    return pairs


def check_kept_database(database: Path, pairs: list[tuple[str, str]]) -> None:
    """Refuse to resume a kept DCTdomain database built from different sequences.

    DCTdomain resumes an existing database and never replaces a sequence it already holds, so
    rerunning ``--keep-intermediate`` after editing a sequence would silently return the old
    protein's fingerprints under the same Entry.
    """
    if not database.is_file():
        return
    try:
        with contextlib.closing(sqlite3.connect(database)) as connection:
            stored = dict(connection.execute("SELECT pid, sequence FROM sequences"))
    except sqlite3.DatabaseError as error:
        raise SystemExit(f"{database} is not a DCTdomain database ({error}); delete {database.parent}") from error
    changed = [entry for entry, sequence in pairs if stored.get(entry, sequence) != sequence]
    if changed:
        raise SystemExit(
            f"{database} was built from different sequences for {len(changed)} of {len(pairs)} "
            f"proteins (first: {changed[0]}). Delete {database.parent} and rerun."
        )


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
    database = workdir / "sequences.db"
    check_kept_database(database, pairs)
    fasta = workdir / "sequences.fasta"
    with fasta.open("w") as handle:
        for entry, sequence in pairs:
            handle.write(f">{entry}\n{sequence}\n")
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
    result = subprocess.run(command, cwd=workdir, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"DCTdomain failed with exit code {result.returncode}; its own error is above. If that "
            f"is an ImportError, it needs torch, fair-esm, scipy and faiss in the interpreter "
            f"running it ({interpreter}): see pyproject.toml, or pass --dct-python."
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
    if len(identifiers) == 0:
        # DCTdomain exits 0 even when its embedding worker died, leaving an empty archive.
        raise SystemExit(f"{path} holds no fingerprints: DCTdomain fingerprinted nothing; its own error is above")
    if len(offsets) != len(identifiers) + 1:
        raise SystemExit(f"{path}: idx must hold one offset per sequence plus a final bound")
    if fingerprints.ndim != 2 or fingerprints.shape[1] != DOMAIN_DIM:
        raise SystemExit(f"{path}: expected {DOMAIN_DIM}-value fingerprints, found {fingerprints.shape}")
    table: dict[str, np.ndarray] = {}
    for position, identifier in enumerate(identifiers):
        if str(identifier) in table:
            # DCTdomain groups consecutive rows by protein; a repeat means they were interleaved.
            raise SystemExit(f"{path}: the fingerprints of {identifier} are not contiguous")
        table[str(identifier)] = fingerprints[offsets[position] : offsets[position + 1], :]
    return table


def check_coverage(pairs: list[tuple[str, str]], fingerprints: dict[str, np.ndarray], path: Path) -> None:
    """Stop before the ESM2 step if DCTdomain left any protein without a fingerprint."""
    missing = [entry for entry, _ in pairs if entry not in fingerprints]
    if missing:
        raise SystemExit(f"{path} has no fingerprint for {len(missing)} of {len(pairs)} proteins (first: {missing[0]})")


def dct_to_representation(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad the DCT fingerprints of one protein into the ``(16, 960)`` bag the models read.

    ``feature`` is what DCTdomain emitted for this protein: one row per domain followed by one
    whole-protein row. Every valid token is ``[its own domain | the whole protein]``, and a protein
    with no domain keeps a single token where both halves are the whole-protein fingerprint.
    """
    dct_count = feature.shape[0] - 1
    representation = np.full((MAX_DOMAINS, 2 * DOMAIN_DIM), 0, dtype=float)
    mask = np.full((MAX_DOMAINS,), False, dtype=bool)
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
        # Shortest-first, so the newcomer is the longest in the batch and sets its padded width.
        longest = len(pairs[position][1]) + 2
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
        representation, mask, dct_count = dct_to_representation(fingerprints[entry])
        if dct_count > MAX_DOMAINS:
            print(f"[warn] {entry}: {dct_count} domains found; keeping the first {MAX_DOMAINS}", flush=True)
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
    parser.add_argument(
        "--dct-gpus", type=int, help="GPUs for DCTdomain embedding; 0 for CPU (default: 1 if CUDA is available)"
    )
    parser.add_argument("--dct-cpus", type=int, default=4, help="CPU workers for DCT fingerprinting")
    parser.add_argument("--dct-maxlen", type=int, default=500, help="DCTdomain sequence-splitting threshold")
    parser.add_argument("--keep-intermediate", action="store_true", help="keep the FASTA and DCTdomain database")
    args = parser.parse_args()

    source = Path(args.csvdata).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"no such file: {source}")
    destination = Path(args.output).expanduser().resolve() if args.output else source.with_suffix(".npz")
    if destination.suffix != ".npz":
        # np.savez appends the suffix itself; do it here so the path reported below is the real one.
        destination = destination.with_name(destination.name + ".npz")
    if args.batch_tokens < 1:
        raise SystemExit("--batch-tokens must be at least one")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] {device} requested but CUDA is unavailable; using CPU")
        device = "cpu"
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dct_gpus = min(available_gpus, 1) if args.dct_gpus is None else args.dct_gpus
    if dct_gpus < 0:
        raise SystemExit("--dct-gpus cannot be negative")
    if dct_gpus > available_gpus:
        print(f"[warn] --dct-gpus {dct_gpus} requested but only {available_gpus} CUDA device(s) found")
        dct_gpus = available_gpus

    pairs = read_sequences(source)
    lengths = [len(sequence) for _, sequence in pairs]
    print(f"{len(pairs)} proteins from {source}, {min(lengths)}-{max(lengths)} residues")

    workdir = None
    try:
        if args.dct_npz:
            fingerprint_path = Path(args.dct_npz).expanduser().resolve()
            if not fingerprint_path.is_file():
                raise SystemExit(f"no such file: {fingerprint_path}")
            print(f"[1/2] reusing DCTdomain fingerprints from {fingerprint_path}")
        else:
            dctdomain = find_dctdomain(args.dctdomain)
            if args.keep_intermediate:
                workdir = destination.parent / f"{destination.stem}_dct"
                workdir.mkdir(parents=True, exist_ok=True)
            else:
                workdir = Path(tempfile.mkdtemp(prefix="enzyme-mil-dct-"))
            fingerprint_path = run_dctdomain(
                pairs, workdir, dctdomain, args.dct_python, dct_gpus, args.dct_cpus, args.dct_maxlen
            )
        fingerprints = read_fingerprints(fingerprint_path)
        check_coverage(pairs, fingerprints, fingerprint_path)
        embeddings = embed_sequences(pairs, device, args.batch_tokens)
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
