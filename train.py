from global_config import *
import os, random
from pathlib import Path
import wandb
from tqdm import tqdm
import clip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTForImageClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from dataloader import MemeData, MemeDataset
from modeling import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=128)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument(
    "--model",
    type=str,
    default="text",
    choices=["text", "img", "multimodal"],
)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--d_hidden", type=int, default=128)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="online",
    choices=["online", "offline", "disabled"],
)
hparams = vars(parser.parse_args())

# set seed
if hparams["seed"] < 0:
    hparams["seed"] = random.randint(0, 9999)


class MemeModel:
    def __init__(
        self,
        data,
        model,
        tokenizer=AutoTokenizer.from_pretrained("albert-base-v2"),
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        accumulate_grad_batches: int = 1,
        max_epochs: int = 10,
        loss_metric=nn.BCEWithLogitsLoss(),
        **kwargs,
    ):
        super().__init__()
        self.data = data
        self.model = model
        self.lr = lr
        self.tokenizer = tokenizer
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_epochs = max_epochs
        self.loss_metric = loss_metric

        self.setup()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)

    def map_device(self, inputs):
        return {k: v.to(DEVICE) for k, v in inputs.items()}

    def setup(self):
        self.model.to(DEVICE)
        self.total_steps = (
            len(self.data.train_loader.dataset)
            // self.data.train_batch_size
            // self.accumulate_grad_batches
            * self.max_epochs
        )

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_ratio * self.total_steps,
            num_training_steps=self.total_steps,
        )

        return optimizer, scheduler


class TextModel(MemeModel):
    def __call__(self, text=None, face_context=None, labels=None):
        text = self.tokenizer(
            face_context,
            text,
            padding="longest",
            return_tensors="pt",
        )
        text, labels = self.map_device(text), labels.to(DEVICE)

        logits = self.model(**text).logits.squeeze()
        labels = labels.squeeze()

        loss = self.loss_metric(logits, labels)
        loss /= self.accumulate_grad_batches

        return loss, logits, labels


class VisualModel(MemeModel):
    def __call__(self, img=None, labels=None):
        img, labels = img.to(DEVICE), labels.to(DEVICE)
        logits = self.model(pixel_values=img).logits.squeeze()
        labels = labels.squeeze()

        loss = self.loss_metric(logits, labels)
        loss /= self.accumulate_grad_batches

        return loss, logits, labels


class MultimodalModel(MemeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip, _ = clip.load("ViT-B/32", DEVICE)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=1
        )

        self.text_model.load_state_dict(torch.load("./best_weights/text.pt"))
        self.text_model = self.text_model.albert.to(DEVICE)
        self.text_model.eval()

    def __call__(
        self,
        img=None,
        text=None,
        face_context=None,
        bow_caption=None,
        bow_celebrity=None,
        labels=None,
        **kwargs,
    ):
        text = self.tokenizer(
            face_context,
            text,
            padding="longest",
            return_tensors="pt",
        )
        text, img, labels = self.map_device(text), img.to(DEVICE), labels.to(DEVICE)
        bow_caption, bow_celebrity = bow_caption.to(DEVICE), bow_celebrity.to(DEVICE)
        with torch.no_grad():
            text = self.text_model(**text).pooler_output
            img = self.clip.encode_image(img).float()

        logits = self.model(img, text, bow_celebrity, bow_caption)
        logits, labels = logits.squeeze(), labels.squeeze()
        loss = self.loss_metric(logits, labels)
        loss /= self.accumulate_grad_batches

        return loss, logits, labels


class Driver:
    def __init__(self, **hparams):
        # Transformer set-up
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=1
        )
        self.visual_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=1
        )
        self.concat_model = ConcatModel(**hparams)

        self.set_model(**hparams)
        self.optimizer, self.scheduler = self.model.configure_optimizers()

    def set_model(self, model="text", **hparams) -> None:
        if model == "text":
            tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
            self.data = MemeData(features=["face_context", "text"], **hparams)
            self.model = TextModel(
                self.data, self.text_model, tokenizer=tokenizer, **hparams
            )
        elif model == "img":
            self.data = MemeData(features=["img"], **hparams)
            self.model = VisualModel(self.data, self.visual_model, **hparams)
        elif model == "multimodal":
            self.data = MemeData(
                features=[
                    "text",
                    "face_context",
                    "img",
                    "bow_caption",
                    "bow_celebrity",
                ],
                **hparams,
            )
            self.model = MultimodalModel(self.data, self.concat_model, **hparams)
        else:
            raise ValueError(f"provided mode '{model}' is not found")

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        n_steps = len(dataloader)
        total_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, desc="train")):
            loss, logits, labels = self.model(**batch)
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % self.model.accumulate_grad_batches == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        return {"loss": total_loss / n_steps * self.model.accumulate_grad_batches}

    def eval_epoch(self, dataloader: DataLoader):
        self.model.eval()
        n_steps = len(dataloader)
        total_loss = 0.0
        total_labels, total_probs, total_preds = [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="eval"):
                loss, logits, labels = self.model(**batch)
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = probs.round()

                total_probs.extend(probs.tolist())
                total_preds.extend(preds.tolist())
                total_labels.extend(labels.tolist())

        return {
            "loss": total_loss / n_steps * self.model.accumulate_grad_batches,
            "preds": total_preds,
            "probs": total_probs,
            "labels": total_labels,
        }

    def train(self):
        dev_loss = []
        dev_roc_auc = []
        dev_accuracy = []

        for epoch in range(self.model.max_epochs):
            train_outputs = self.train_epoch(self.data.train_loader)
            dev_outputs = self.eval_epoch(self.data.dev_loader)

            dev_loss.append(dev_outputs["loss"])
            dev_roc_auc.append(
                roc_auc_score(dev_outputs["labels"], dev_outputs["probs"])
            )
            dev_accuracy.append(
                accuracy_score(dev_outputs["labels"], dev_outputs["preds"])
            )

            if max(dev_roc_auc) == dev_roc_auc[epoch]:
                self.model.save(f"./weights/{wandb.run.id}.pt")

            wandb.log(
                {
                    "train_loss": train_outputs["loss"],
                    "dev_loss": dev_loss[epoch],
                    "best_dev_loss": min(dev_loss),
                    "dev_accuracy": dev_accuracy[epoch],
                    "best_dev_accuracy": max(dev_accuracy),
                    "dev_roc_auc": dev_roc_auc[epoch],
                    "best_dev_roc_auc": max(dev_roc_auc),
                }
            )


if __name__ == "__main__":
    wandb.init(project="meme-pretrain", mode=hparams["wandb_mode"])
    wandb.config.update(hparams)

    set_random_seed(hparams["seed"])
    driver = Driver(**hparams)
    driver.train()
