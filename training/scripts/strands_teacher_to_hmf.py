"""
Converts a Strands Decider teacher file to an HMF teacher file (decision_teacher_file).

Strands' teacher files, such as data/synthetic/replay_v14_multistep.jsonl, have one line per row:

    {"i": 112456, "probs": [0.82, 0.11, 0.07]}

where i is the row's position in the concatenation of the training files Strands lists, in order
(train_v5.jsonl, then multistep_v14.jsonl, for replay_v14_multistep.jsonl). HMF's teacher files
name each row by its dataset and its position in that dataset's file instead:

    {"source": "strands-decider/multistep_v14.jsonl", "row_index": 12, "probs": [0.82, 0.11, 0.07]}

Both list probs in the row's original option order, so they are copied unchanged.

    python scripts/strands_teacher_to_hmf.py \
        --teacher path/to/strands-decider/data/synthetic/replay_v14_multistep.jsonl \
        --datasets strands_train_v5,strands_multistep_v14 \
        --output path/to/replay_v14_multistep.hmf.jsonl

--datasets are the HMF datasets (dataset_info.json) of the files i counts over, in the same order.
"""

import json
import os

import fire

from hmf.data.parser import get_dataset_list


def _count_rows(path: str) -> int:
    """The number of rows of a JSON Lines file, which is its number of non-empty lines (as HMF and Strands count)."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def convert(teacher: str, datasets: str, output: str, dataset_dir: str = "data"):
    attrs = get_dataset_list([name.strip() for name in datasets.split(",")], dataset_dir)
    # The range of i covered by each dataset's file
    ranges = []
    start = 0
    for attr in attrs:
        num_rows = _count_rows(os.path.join(dataset_dir, attr.dataset_name))
        ranges.append((attr.dataset_name, start, start + num_rows))
        start += num_rows

    counts = {source: 0 for source, _, _ in ranges}
    with open(teacher, encoding="utf-8") as f, open(output, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            i = int(entry["i"])
            for source, first, end in ranges:
                if first <= i < end:
                    out.write(json.dumps({"source": source, "row_index": i - first, "probs": entry["probs"]}) + "\n")
                    counts[source] += 1
                    break
            else:
                raise ValueError(f"{teacher}: i = {i} is beyond the {start:,} rows of {datasets}.")

    for source, count in counts.items():
        print(f"{source}: {count:,} rows")
    print(f"Written to {output}")


if __name__ == "__main__":
    fire.Fire(convert)
