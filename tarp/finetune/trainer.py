"""Fine-tune a sequence-classification model (LoRA last-3 by default) via HF Trainer.

Returns the fine-tuned (peft) model plus test accuracy/logits. The model is later wrapped
by ``HFTextEncoder`` to extract post-FT representations. Caching of the adapter is handled
one level up (``pipeline``), keyed by ``RunConfig.finetune_key()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from peft import get_peft_model
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from tarp.config import RunConfig
from tarp.data.loaders import TaskData
from tarp.encoders.hf_text import HFTextEncoder
from tarp.finetune.lora import build_lora_config


@dataclass
class FinetuneResult:
    model: object          # fine-tuned (peft) seq-cls model
    tokenizer: object
    test_acc: float
    test_logits: np.ndarray
    val_acc: float | None


class _TextClsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = [int(y) for y in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["labels"] = self.labels[i]
        return item


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def finetune(encoder: HFTextEncoder, task: TaskData, cfg: RunConfig, output_dir) -> FinetuneResult:
    tok = encoder.tokenizer
    model = encoder.classification_model(task.num_labels)
    if cfg.finetune.regime == "lora_last3":
        model = get_peft_model(model, build_lora_config(model, cfg.finetune))
    elif cfg.finetune.regime != "full":
        raise ValueError(f"unknown finetune regime '{cfg.finetune.regime}'")

    ds_tr = _TextClsDataset(task.train[0], task.train[1], tok, cfg.max_length)
    ds_va = _TextClsDataset(task.val[0], task.val[1], tok, cfg.max_length)
    ds_te = _TextClsDataset(task.test[0], task.test[1], tok, cfg.max_length)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.finetune.epochs,
        per_device_train_batch_size=cfg.finetune.batch_size,
        per_device_eval_batch_size=128,
        learning_rate=cfg.finetune.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=cfg.seed,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    val_metrics = trainer.evaluate()
    val_acc = float(val_metrics["eval_accuracy"]) if "eval_accuracy" in val_metrics else None

    logits = trainer.predict(ds_te).predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    test_acc = float((np.argmax(logits, axis=-1) == np.asarray(task.test[1])).mean())

    return FinetuneResult(
        model=model, tokenizer=tok, test_acc=test_acc, test_logits=logits, val_acc=val_acc
    )
