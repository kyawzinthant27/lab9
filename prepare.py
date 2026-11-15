import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from lab9 import CONFIG


def main():
    folder = Path("data_notebook")
    folder.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["TOKENIZER_NAME"]
    )
    tokenizer.save_pretrained(folder / "tokenizer")

    sizes = {
        "train": CONFIG["TRAIN_SIZE"],
        "validation": CONFIG["VALID_SIZE"],
        "test": CONFIG["TEST_SIZE"],
    }

    info = {
        "dataset": "abisee/cnn_dailymail",
        "config": CONFIG,
        "splits": {},
    }

    for split, size in sizes.items():
        print(f"Preparing {split}...", flush=True)

        url = (
            "https://huggingface.co/api/datasets/"
            "abisee/cnn_dailymail/parquet/"
            f"{CONFIG['DATASET_VERSION']}/{split}/0.parquet"
        )

        data = pd.read_parquet(
            url,
            columns=["article", "highlights", "id"],
        )

        data = data.dropna().drop_duplicates("id")

        data = data[
            data["article"].str.strip().ne("")
            & data["highlights"].str.strip().ne("")
        ]

        data = data.sample(
            n=size,
            random_state=CONFIG["SEED"],
        )

        source = tokenizer(
            data["article"].tolist(),
            truncation=True,
            max_length=CONFIG["MAX_INPUT_LENGTH"],
        )["input_ids"]

        target = tokenizer(
            data["highlights"].tolist(),
            truncation=True,
            max_length=CONFIG["MAX_TARGET_LENGTH"],
        )["input_ids"]

        examples = [
            {"input_ids": x, "labels": y}
            for x, y in zip(source, target)
        ]

        with (folder / f"{split}.json").open("w") as file:
            json.dump(examples, file)

        info["splits"][split] = {
            "count": size,
            "ids": data["id"].tolist(),
        }

        if split == "test":
            data = data.rename(
                columns={
                    "article": "source",
                    "highlights": "target",
                }
            )

            data.to_json(
                folder / "test_text.json",
                orient="records",
                force_ascii=False,
            )

        print(f"Saved {size} examples.", flush=True)

    with (folder / "preprocessing.json").open("w") as file:
        json.dump(info, file, indent=2)

    print("Preparation finished.")


if __name__ == "__main__":
    main()
