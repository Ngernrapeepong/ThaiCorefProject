import argparse
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.dataset import load_coref_data, prepare_coref_examples
from src.model import ZCorefModel


DEFAULT_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Thai mention detection and coreference model."
    )
    parser.add_argument("--train-data", default="data/train.json")
    parser.add_argument("--dev-data", default="data/dev.json")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/thai-coref")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _tensorize(example, device):
    input_ids = torch.tensor(
        [example["input_ids"]], dtype=torch.long, device=device
    )
    attention_mask = torch.tensor(
        [example["attention_mask"]], dtype=torch.long, device=device
    )
    mentions = torch.tensor(
        [mention[:2] for mention in example["mentions"]],
        dtype=torch.long,
        device=device,
    ).reshape(-1, 2)
    cluster_ids = [mention[2] for mention in example["mentions"]]
    return input_ids, attention_mask, mentions, cluster_ids


def _coreference_loss(antecedent_scores, cluster_ids):
    mention_count = len(cluster_ids)
    if mention_count == 0:
        return antecedent_scores.sum()

    losses = []
    for mention_index, cluster_id in enumerate(cluster_ids):
        gold_antecedents = [
            previous_index + 1
            for previous_index in range(mention_index)
            if cluster_ids[previous_index] == cluster_id
        ]
        if not gold_antecedents:
            gold_antecedents = [0]

        candidate_scores = antecedent_scores[mention_index]
        gold_scores = candidate_scores[gold_antecedents]
        losses.append(
            torch.logsumexp(candidate_scores, dim=0)
            - torch.logsumexp(gold_scores, dim=0)
        )
    return torch.stack(losses).mean()


def _document_loss(model, example, device):
    input_ids, attention_mask, mention_spans, cluster_ids = _tensorize(
        example, device
    )
    start_scores, end_scores, antecedent_scores = model(
        input_ids, attention_mask, mention_spans
    )
    start_logits = start_scores[0, :, 0]
    end_logits = end_scores[0, :, 0]
    start_targets = torch.zeros_like(start_logits)
    end_targets = torch.zeros_like(end_logits)
    if mention_spans.numel():
        start_targets[mention_spans[:, 0]] = 1.0
        end_targets[mention_spans[:, 1]] = 1.0

    def boundary_loss(logits, targets):
        positive_count = targets.sum()
        negative_count = targets.numel() - positive_count
        positive_weight = (negative_count / positive_count.clamp(min=1)).clamp(
            min=1.0, max=20.0
        )
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=positive_weight
        )

    mention_loss = (
        boundary_loss(start_logits, start_targets)
        + boundary_loss(end_logits, end_targets)
    ) / 2
    coreference_loss = _coreference_loss(antecedent_scores, cluster_ids)
    return mention_loss + coreference_loss


def _evaluate(model, examples, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for example in examples:
            total_loss += _document_loss(model, example, device).item()
    return total_loss / max(len(examples), 1)


def _save_checkpoint(model, tokenizer, output_dir, model_name, epoch, score,
                     max_length, overlap):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "epoch": epoch,
            "loss": score,
            "max_length": max_length,
            "overlap": overlap,
        },
        os.path.join(output_dir, "checkpoint.pt"),
    )
    tokenizer.save_pretrained(output_dir)


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs ต้องมีค่าอย่างน้อย 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate ต้องมากกว่า 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_records = load_coref_data(args.train_data)
    if not train_records:
        raise ValueError(f"ไม่พบข้อมูลฝึกใน {args.train_data}")
    dev_records = load_coref_data(args.dev_data) if args.dev_data else []

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("ต้องใช้ fast tokenizer เพื่ออ่าน character offsets")
    model = ZCorefModel(args.model_name)
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and args.max_length > max_positions:
        raise ValueError(
            f"--max-length ({args.max_length}) มากกว่าโมเดลรองรับ "
            f"({max_positions})"
        )

    train_examples = prepare_coref_examples(
        train_records, tokenizer, args.max_length, args.overlap
    )
    dev_examples = prepare_coref_examples(
        dev_records, tokenizer, args.max_length, args.overlap
    ) if dev_records else []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")

    print(
        f"เอกสารฝึก {len(train_records)} รายการ, "
        f"หน้าต่างฝึก {len(train_examples)} ชุด, device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_examples)
        total_loss = 0.0
        for example in train_examples:
            optimizer.zero_grad(set_to_none=True)
            loss = _document_loss(model, example, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_examples)
        score = _evaluate(model, dev_examples, device) if dev_examples else train_loss
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss={train_loss:.4f}"
            + (f" - dev_loss={score:.4f}" if dev_examples else "")
        )
        if score < best_loss:
            best_loss = score
            _save_checkpoint(
                model,
                tokenizer,
                args.output_dir,
                args.model_name,
                epoch,
                score,
                args.max_length,
                args.overlap,
            )

    print(f"บันทึก checkpoint แล้วที่ {args.output_dir}")


if __name__ == "__main__":
    main()
