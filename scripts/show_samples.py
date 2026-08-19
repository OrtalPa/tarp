"""Show true dataset sizes + 10 shuffled RAW rows from each dataset.

Unlike peek_datasets.py (which streamed the first few rows), this downloads the
full dataset so we can report real split sizes and sample across all labels.
"""

from datasets import load_dataset, load_dataset_builder

SPECS = [
    ("SST-2",     ("stanfordnlp/sst2",),  {}),
    ("AG News",   ("fancyzhx/ag_news",),  {}),
    ("TREC",      ("SetFit/TREC-QC",),    {}),
    ("Emotion",   ("dair-ai/emotion",),   {}),
    ("Banking77", ("mteb/banking77",),    {}),
    ("CLINC150",  ("clinc/clinc_oos",),   {"name": "plus"}),
]

N = 10


def main():
    for name, args, kwargs in SPECS:
        print("=" * 90)
        print(f"### {name}   load_dataset{args} {kwargs}")
        # True sizes without assuming anything — straight from the builder info.
        b = load_dataset_builder(*args, **kwargs)
        splits = b.info.splits or {}
        sizes = ", ".join(f"{k}={v.num_examples}" for k, v in splits.items())
        print(f"    splits: {sizes}")

        ds = load_dataset(*args, split="train", **kwargs)
        print(f"    columns: {ds.column_names}   (train rows: {len(ds)})")
        print(f"    --- {N} shuffled raw rows ---")
        for row in ds.shuffle(seed=0).select(range(N)):
            clean = {k: (str(v).replace(chr(10), " ")) for k, v in row.items()}
            print("   ", clean)
        print()


if __name__ == "__main__":
    main()
