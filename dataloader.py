import pandas as pd
import random
from ast import literal_eval
from typing import List
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter


class MemeDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, features: List[str], fold: str = "train"
    ) -> None:
        self.df, self.fold = df, fold
        self.length = len(df)
        self.features = features
        self.meme_stats = (0.5147, 0.4849, 0.4632), (0.3126, 0.3080, 0.3147)
        self.imagenet_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.clip_stats = (0.48145466, 0.4578275, 0.40821073), (
            0.26862954,
            0.26130258,
            0.27577711,
        )
        self.transform = T.Compose(
            [
                T.Resize(224, interpolation=Image.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(*self.clip_stats),
            ]
        )

    def __len__(self):
        return self.length

    # AVAILABLE FEATURES:
    # ===TEXT===
    # text (str): meme text
    # faces (List[str]): facial metadata text
    # celebrity_name (List[str]): name of celebrity in image
    # celebrity_keywords (List[str]): keywords related to celebrity
    # celebrtity_summary (List[str]): 1~2 paragraph summary of celebrity
    # captions: image caption
    # ... "caption_text", "face_text", "face_context"
    # ===IMAGE===
    # img: 224 X 224 X 3 image
    # img_grid: 4 X 4 image grid

    def __getitem__(self, idx: int):
        id = self.df.index[idx]
        instance = {}

        if self.fold in ["train", "dev"]:
            instance["labels"] = float(self.df.loc[id, "label"])

        for f in self.features:
            instance[f] = self.get_feature(id, f)

        return instance

    # ALL AVAILABLE FEATURES
    def get_feature(self, id: str, feature: str):
        if feature in ["text", "bow_caption", "bow_celebrity"]:
            return self.df.loc[id, feature]
        elif feature == "caption":
            return self.get_caption(id)
        elif feature == "caption_text":
            return self.get_text(id, text_list=["IMAGE", "TEXT"])
        elif feature == "face_text":
            return self.get_text(id, text_list=["NAME", "FACE", "IMAGE", "TEXT"])
        elif feature == "face_context":
            return self.get_text(id, text_list=["NAME", "FACE", "IMAGE"])
        elif feature == "img":
            return self.get_img(id)
        elif feature == "img_grid":
            return self.get_img_grid(id)
        else:
            raise ValueError(f"Invalid feature {feature}")

    def get_text(self, id: str, text_list=["NAME", "FACE", "IMAGE", "TEXT"]):
        join = lambda l: ". ".join(l)
        text_map = {
            "TEXT": self.df.loc[id, "text"],
            "IMAGE": self.get_caption(id),
            "FACE": join(self.df.loc[id, "faces"]),
            "NAME": join(self.df.loc[id, "celebrity_name"]),
        }

        text = [(t + ": " + text_map[t]) for t in text_list]
        text = " ".join(text)

        return text

    def get_caption(self, id: str):
        captions = self.df.loc[id, "captions"]
        caption = random.choice(captions) if self.fold == "train" else captions[0]

        return caption

    def get_grid_bbox(self, id: str, scale: int) -> List[List[int]]:
        width, height = (
            self.df.loc[id, "image_w"] // scale,
            self.df.loc[id, "image_h"] // scale,
        )
        bbox = []
        for i in range(scale):
            for j in range(scale):
                bbox.append([i * width, j * height, (i + 1) * width, (j + 1) * height])

        return bbox

    def extract_boxes(self, id: str, bbox: List[List[int]], max_length: int = 1):
        null_crops = torch.zeros((max_length, 224, 224))
        if len(bbox) == 0:
            return null_crops

        try:
            img = Image.open(f"./data/img/{id}.png").convert("RGB")
            crops = [self.transform(img.crop(bounds)) for bounds in bbox[:max_length]]
            return crops
        except:
            return null_crops

    def get_img(self, id: str):
        img = Image.open(f"./data/img/{id}.png").convert("RGB")
        return self.transform(img)

    def get_img_grid(self, id: str):
        grid_bbox = self.get_grid_bbox(id, 4)
        return self.extract_boxes(id, grid_bbox, max_length=4 * 4)


class MemeData:
    def __init__(
        self,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        num_workers: int = 4,
        face_meta_threshold=90,
        gender_threshold=95,
        celebrity_threshold=2,
        race_threshold=95,
        celebrity_match_threshold=95,
        face_confidence_threshold=95,
        max_faces=4,
        features=["img"],
        **kwargs,
    ):
        self.data_folder = Path("data")
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.face_meta_threshold = face_meta_threshold
        self.gender_threshold = gender_threshold
        self.celebrity_threshold = celebrity_threshold
        self.race_threshold = race_threshold
        self.celebrity_match_threshold = celebrity_match_threshold
        self.face_confidence_threshold = face_confidence_threshold
        self.max_faces = max_faces
        self.features = features
        self.dataset = {}

        self.setup()

    def align_df_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df["id"] = df["id"].apply(lambda id: str(id).zfill(5))
        return df

    def get_captions(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_folder / "captions.csv")
        df = self.align_df_id(df)
        df["captions"] = df["captions"].apply(literal_eval)

        return df

    def get_celebrity_meta(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_folder / "faces/description.csv")
        df = self.align_df_id(df)
        wiki = pd.read_csv(self.data_folder / "faces/wikipedia.csv")
        df = df.merge(wiki, on="name")

        # drop celebrity with low counts
        counter = Counter(df["name"])
        has_low_count = df["name"].apply(
            lambda name: counter.get(name) <= self.celebrity_threshold
        )

        df = df[
            ~has_low_count
            & ~df["name"].isna()
            & (df["confidence"] > self.face_confidence_threshold)
            & (df["match_confidence"] > self.celebrity_match_threshold)
        ]
        df["keywords"] = df["keywords"].fillna("")
        df["summary"] = df["summary"].fillna("")

        return df[["id", "name", "keywords", "summary"]]

    def get_age_group(self, age: int) -> str:
        assert age >= 0, f"invalid age {age}"
        if age < 14:
            return "child"
        elif 14 <= age < 25:
            return "young"
        elif 25 <= age < 40:
            return "adult"
        elif 40 <= age < 60:
            return "middle-age"
        else:
            return "old"

    def get_face_meta(self) -> pd.DataFrame:

        facial_features = [
            "smile",
            "eyeglasses",
            "sunglasses",
            "beard",
            "mustache",
            # "eyesopen",
            # "mouthopen",
        ]

        race_list = [
            "asian",
            "indian",
            "black",
            "white",
            "middle eastern",
            "latino hispanic",
        ]

        emotion_list = [
            "CALM",
            "SAD",
            "ANGRY",
            "CONFUSED",
            "HAPPY",
            "FEAR",
            "DISGUSTED",
            "SURPRISED",
        ]

        df = pd.read_csv(self.data_folder / "faces/general.csv")
        df = self.align_df_id(df)
        df = df[df["confidence"] > self.face_confidence_threshold]
        df["face_meta"] = ""

        # add gender
        df["gender"] = df.apply(
            lambda entry: entry["gender_value"] + " "
            if entry["gender_confidence"] > self.gender_threshold
            else "",
            axis="columns",
        )

        # add race
        df["race"] = df[race_list].idxmax(axis="columns")
        df["face_meta"] += df.apply(
            lambda entry: entry["race"] + " "
            if entry[entry["race"]] > self.race_threshold
            else "",
            axis="columns",
        )

        # add age
        df["age"] = (df["agerange_low"] + df["agerange_high"]) / 2
        df["face_meta"] += df["age"].apply(self.get_age_group) + " "

        # add emotion
        for emotion in emotion_list:
            df["face_meta"] += df.apply(
                lambda entry: emotion.lower() + " "
                if entry[f"{emotion}_confidence"] > self.face_meta_threshold
                else "",
                axis="columns",
            )

        # add facial features
        for feature in facial_features:
            df["face_meta"] += df.apply(
                lambda entry: feature + " "
                if entry[f"{feature}_confidence"] > self.face_meta_threshold
                and entry[f"{feature}_value"]
                else "",
                axis="columns",
            )

        return df[["id", "face_meta"]]

    def get_object(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_folder / "objects.csv")
        df = self.align_df_id(df)

        df["box_labels"] = df["box_labels"].apply(literal_eval)
        df["bbox"] = df["bbox"].apply(literal_eval)

        df = df[df["num_boxes"] > 0]
        df = df[(df["image_w"] > 100) & (df["image_h"] > 100)]

        df["area"] = df.apply(
            lambda entry: entry["image_w"] * entry["image_h"], axis="columns"
        )
        df["proportion"] = df.apply(
            lambda entry: [
                (l - r) * (t - b) / entry["area"] for l, t, r, b in entry["bbox"]
            ],
            axis="columns",
        )
        df["box_labels"] = df.apply(
            lambda entry: [
                entry["box_labels"][idx]
                for idx, p in enumerate(entry["proportion"])
                if p > 0.05
            ],
            axis="columns",
        )
        df["box_labels"] = df["box_labels"].apply(lambda l: ", ".join(l))
        df["bbox"] = df.apply(
            lambda entry: [
                entry["bbox"][idx]
                for idx, p in enumerate(entry["proportion"])
                if p > 0.05
            ],
            axis="columns",
        )

        return df

    def get_bow(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_folder / "BoW.csv")
        df = self.align_df_id(df)
        to_tensor = lambda l: torch.FloatTensor(literal_eval(l))
        df["bow_caption"] = df["bow_caption"].apply(to_tensor)
        df["bow_celebrity"] = df["bow_celebrity"].apply(to_tensor)

        return df

    def setup(self):
        # aggregate and set index to id
        captions = self.get_captions().set_index("id")
        bow = self.get_bow().set_index("id")
        face = self.get_face_meta().groupby("id").agg(list)
        celebrity_meta = self.get_celebrity_meta().groupby("id").agg(list)

        for fold in ["train", "dev", "test"]:
            df = pd.read_csv(self.data_folder / f"{fold}.csv")
            df = self.align_df_id(df).set_index("id").drop_duplicates()
            df = df.join(captions).join(face).join(celebrity_meta).join(bow)

            # fill NaN with empty list and join
            fillna = lambda d: d if isinstance(d, list) else []
            df["faces"] = df["face_meta"].apply(fillna)
            df["celebrity_name"] = df["name"].apply(fillna)
            df["celebrity_keywords"] = df["keywords"].apply(fillna)
            df["celebrity_summary"] = df["summary"].apply(fillna)

            # fill NaN caption
            df["captions"] = df["captions"].fillna("")
            # trim face (in case of group of people)
            df["faces"] = df["faces"].apply(lambda l: l[: self.max_faces])

            self.dataset[fold] = MemeDataset(df, self.features, fold)
            save_columns = [
                "img",
                "text",
                "captions",
                "faces",
                "celebrity_name",
                "celebrity_keywords",
                "celebrity_summary",
                "bow_caption",
                "bow_celebrity",
            ]
            df[save_columns].to_csv(f"./data/tensors/{fold}.csv")

        # init data loader
        self.train_loader = self.train_dataloader()
        self.dev_loader = self.dev_dataloader()
        self.test_loader = self.test_dataloader()

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def dev_dataloader(self):
        return DataLoader(
            self.dataset["dev"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dataset = MemeData(
        train_batch_size=8,
        eval_batch_size=8,
        features=["text", "bow_caption", "bow_celebrity"],
    )
    print(next(iter(dataset.train_loader)))
