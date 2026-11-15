import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from rouge_score import rouge_scorer

from transformer import (
    Encoder,
    Decoder,
    Generator,
    Transformer,
    NoamOptim,
)


CONFIG = {
    "SEED": 42,
    "TOKENIZER_NAME": "bert-base-multilingual-cased",
    "DATASET_VERSION": "1.0.0",
    "TRAIN_SIZE": 10000,
    "VALID_SIZE": 500,
    "TEST_SIZE": 200,
    "MAX_INPUT_LENGTH": 512,
    "MAX_TARGET_LENGTH": 128,
}

BATCH_SIZE = 8
EPOCHS = 10

DATA = Path("data_notebook")
RESULTS = Path("results_notebook")


def load_data(split, tokenizer):
    with (DATA / f"{split}.json").open() as file:
        examples = json.load(file)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=tokenizer.pad_token_id,
        return_tensors="pt",
    )

    return DataLoader(
        examples,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        collate_fn=collator,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        source = batch["input_ids"].to(device)
        target = batch["labels"].to(device)

        with torch.set_grad_enabled(optimizer is not None):
            output, _ = model(source, target[:, :-1])
            expected = target[:, 1:]

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                expected.reshape(-1),
            )

            tokens = expected.ne(model.pad_idx).sum().item()

            if optimizer is not None:
                optimizer.zero_grad()
                (loss / tokens).backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        total_tokens += tokens

    return total_loss / total_tokens


@torch.no_grad()
def summarize(model, source, tokenizer):
    model.eval()

    src_mask = model.get_pad_mask(source)
    memory = model.encoder(source, src_mask)

    target = torch.full(
        (source.size(0), 1),
        tokenizer.cls_token_id,
        dtype=torch.long,
        device=source.device,
    )

    finished = torch.zeros(
        source.size(0),
        dtype=torch.bool,
        device=source.device,
    )

    for _ in range(CONFIG["MAX_TARGET_LENGTH"] - 1):
        tgt_mask = (
            model.get_pad_mask(target)
            & model.get_subsequent_mask(target)
        )

        output, _ = model.decoder(
            target, memory, src_mask, tgt_mask
        )

        scores = model.generator(output[:, -1])
        scores[:, [model.pad_idx, tokenizer.cls_token_id]] = -float("inf")

        token = scores.argmax(dim=-1)
        token = token.masked_fill(finished, model.pad_idx)

        target = torch.cat([target, token.unsqueeze(1)], dim=1)
        finished |= token.eq(tokenizer.sep_token_id)

        if finished.all():
            break

    return tokenizer.batch_decode(
        target.cpu().tolist(),
        skip_special_tokens=True,
    )


def main():
    torch.manual_seed(CONFIG["SEED"])

    if not torch.cuda.is_available():
        raise RuntimeError("Run this file through the GPU batch job.")

    device = torch.device("cuda")
    RESULTS.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        DATA / "tokenizer",
        local_files_only=True,
    )

    train = load_data("train", tokenizer)
    validation = load_data("validation", tokenizer)
    test = load_data("test", tokenizer)

    vocabulary = len(tokenizer)
    pad = tokenizer.pad_token_id

    encoder = Encoder(vocabulary, 512, 6, 8, 2048, pad)
    decoder = Decoder(vocabulary, 512, 6, 8, 2048, pad)
    generator = Generator(512, vocabulary)

    model = Transformer(
        encoder, decoder, generator, pad
    ).to(device)

    adam = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    optimizer = NoamOptim(adam, 512, 2, 4000)
    criterion = nn.NLLLoss(ignore_index=pad, reduction="sum")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Training examples:", len(train.dataset))
    print("Validation examples:", len(validation.dataset))
    print("Test examples:", len(test.dataset))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    best_loss = float("inf")
    checkpoint = RESULTS / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, train, criterion, device, optimizer
        )

        valid_loss = run_epoch(
            model, validation, criterion, device
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train loss={train_loss:.4f}, "
            f"validation loss={valid_loss:.4f}",
            flush=True,
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    )

    test_loss = run_epoch(model, test, criterion, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {math.exp(test_loss):.4f}")
    print("Generating summaries...", flush=True)

    predictions = []

    for batch in test:
        source = batch["input_ids"].to(device)
        predictions.extend(summarize(model, source, tokenizer))

    with (DATA / "test_text.json").open() as file:
        records = json.load(file)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    assert len(records) == len(predictions)

    for record, prediction in zip(records, predictions):
        scores = scorer.score(record["target"], prediction)

        for name in totals:
            totals[name] += scores[name].fmeasure

        record["prediction"] = prediction

    metrics = {
        "test_loss": test_loss,
        "test_perplexity": math.exp(test_loss),
    }

    for name, total in totals.items():
        metrics[name] = 100 * total / len(records)
        print(f"{name}: {metrics[name]:.2f}")

    with (RESULTS / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    with (RESULTS / "summaries.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    for record in records[:3]:
        print("\nReference:", record["target"])
        print("Generated:", record["prediction"])

    print("\nFinished.")


if __name__ == "__main__":
    main()
