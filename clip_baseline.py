from global_config import *
import wandb
import torch
import clip

import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from dataloader import MemeData, MemeDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--C", type=float, default=1)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="online",
    choices=["online", "offline", "disabled"],
)

model, preprocess = clip.load("ViT-B/32")
hparams = vars(parser.parse_args())

# set seed
if hparams["seed"] < 0:
    hparams["seed"] = random.randint(0, 9999)


def get_features(loader, cache_path: Path = None):
    all_features = []
    all_labels = []

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    with torch.no_grad():
        for batch in tqdm(loader):
            img, labels = batch["img"], batch["labels"]
            features = model.encode_image(img.to(DEVICE))

            all_features.append(features)
            all_labels.append(labels)

    result = {
        "features": torch.cat(all_features).cpu().numpy(),
        "labels": torch.cat(all_labels).cpu().numpy(),
    }

    with open(cache_path, "wb") as f:
        pkl.dump(result, f)

    return result


def train(C: float, seed: int):
    data = MemeData(features=["img"], train_batch_size=256, eval_batch_size=256)
    train_data_path = Path("./data/tensors/train_clip.pkl")
    dev_data_path = Path("./data/tensors/dev_clip.pkl")

    train_data = get_features(data.train_loader, cache_path=train_data_path)
    dev_data = get_features(data.dev_loader, cache_path=dev_data_path)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=seed, C=C, max_iter=1000)
    classifier.fit(train_data["features"], train_data["labels"])

    # Evaluate using the logistic regression classifier
    preds = classifier.predict(dev_data["features"])
    probs = classifier.predict_proba(dev_data["features"])
    probs = probs[:, 1]

    wandb.log(
        {
            "dev_accuracy": accuracy_score(dev_data["labels"], preds),
            "dev_roc_auc": roc_auc_score(dev_data["labels"], probs),
        }
    )


wandb.init(project="meme-pretrain", mode=hparams["wandb_mode"])
wandb.config.update(hparams)

set_random_seed(hparams["seed"])
train(hparams["C"], hparams["seed"])
