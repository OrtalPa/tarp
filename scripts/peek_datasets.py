"""Peek at a few examples from each text dataset in the study.

Uses streaming so we don't download full datasets just to look. Prints the label
space and 3 sample rows per dataset.
"""

from itertools import islice

from datasets import load_dataset

# (display name, load_dataset args, kwargs, text field, label field for display)
# NOTE: datasets>=3.0 dropped script-based datasets, so we use parquet-native mirrors.
SPECS = [
    ("SST-2",     ("stanfordnlp/sst2",),  {},               "sentence", "label"),
    ("AG News",   ("fancyzhx/ag_news",),  {},               "text",     "label"),
    ("TREC",      ("SetFit/TREC-QC",),    {},               "text",     "label_coarse_text"),
    ("Emotion",   ("dair-ai/emotion",),   {},               "text",     "label"),
    ("Banking77", ("mteb/banking77",),    {},               "text",     "label_text"),
    ("CLINC150",  ("clinc/clinc_oos",),   {"name": "plus"}, "text",     "intent"),
]

N = 3


def label_names(ds, label_field):
    try:
        feat = ds.features[label_field]
        names = getattr(feat, "names", None)
        if names:
            return names
    except Exception:
        pass
    return None


def main():
    for name, args, kwargs, text_field, label_field in SPECS:
        print("=" * 78)
        print(f"### {name}   load_dataset{args} {kwargs}")
        try:
            ds = load_dataset(*args, split="train", streaming=True, **kwargs)
            names = label_names(ds, label_field)
            n_classes = len(names) if names else "?"
            print(f"    label field: '{label_field}'   #classes: {n_classes}")
            if names and len(names) <= 12:
                print(f"    classes: {names}")
            print(f"    text field:  '{text_field}'")
            print("    --- samples ---")
            for row in islice(ds, N):
                lab = row.get(label_field)
                lab_name = names[lab] if (names and isinstance(lab, int)) else lab
                txt = str(row.get(text_field, "")).replace("\n", " ")
                if len(txt) > 160:
                    txt = txt[:160] + "…"
                print(f"    [{lab_name}]  {txt}")
        except Exception as e:
            print(f"    !! FAILED: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
