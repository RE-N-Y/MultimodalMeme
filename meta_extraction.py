import os
import json
import boto3
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import requests
from ast import literal_eval
from deepface import DeepFace
from ISR.models import RRDN

rrdn = RRDN(weights="gans")

with open("secrets.json") as f:
    secrets = json.load(f)

base = Path("./data")
os.environ["AWS_ACCESS_KEY_ID"] = secrets["key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = secrets["secret"]
client = boto3.client("rekognition")


def run_aws(uid: str, process, out_folder: Path):
    in_file = base / f"img/{uid}.png"
    out_file = out_folder / f"{uid}.json"

    if out_file.exists():
        return

    with open(in_file, "rb") as f:
        response = process(Image={"Bytes": f.read()}, Attributes=["ALL"])

    with open(out_file, "w") as f:
        json.dump(response, f)


def extract_aws(process, out_folder: Path):
    img_folder = base / "img"
    fail = []

    for file in tqdm(img_folder.iterdir()):
        try:
            run_aws(file.stem, process, out_folder)
        except Exception as e:
            print("failure on: ", file)
            print("error: ", e)
            fail.append(file)

    with open("./fail.log", "w") as f:
        f.write(str(fail))


def get_knowledge_graph(query: str):
    uri = "https://kgsearch.googleapis.com/v1/entities:search"

    params = {
        "query": query,
        "key": secrets["kg_api_key"],
    }
    response = requests.get(uri, params=params)
    content = json.loads(response.content)

    return content


def get_face_id() -> pd.DataFrame:
    face_id = []
    face_folder = base / "face_id"

    for file in face_folder.iterdir():
        with open(file, "r") as f:
            data = json.load(f)
        data["id"] = int(file.stem)
        face_id.append(data)

    face_id = pd.DataFrame(face_id)

    return face_id


def extract_name(celebrity_list: list) -> list:
    return [celebrity["Name"] for celebrity in celebrity_list]


def extract_knowledge_graph():
    face_id = get_face_id()
    face_id["CelebrityNames"] = face_id["CelebrityFaces"].apply(
        lambda l: extract_name(l)
    )
    face_id["CelebrityDescriptions"] = face_id["CelebrityNames"].apply(
        lambda l: [get_knowledge_graph(name) for name in l]
    )

    face_id.to_csv(base / "face_id.csv", index=False)


def get_info(result, keys: list):
    try:
        for key in keys:
            result = result[key]
        return result
    except:
        return ""


def get_summary(meta_list: list, key: str):
    meta_summary = [meta["name"] + " : " + meta[key] for meta in meta_list]
    return meta_summary


def get_description(celebrity):
    if len(celebrity["itemListElement"]) == 0:
        return {"name": "", "description": "", "detail": ""}

    top_search = max(celebrity["itemListElement"], key=lambda p: p["resultScore"])
    result = top_search["result"]

    description = get_info(result, ["description"])
    detail = get_info(result, ["detailedDescription", "articleBody"])
    name = get_info(result, ["name"])

    return {"name": name, "description": description, "detail": detail}


def add_celebrity_description():
    face_id = pd.read_csv(base / "face_id.csv")
    face_id["CelebrityNames"] = face_id["CelebrityNames"].apply(literal_eval)
    celebrity_description = face_id["CelebrityDescriptions"].apply(literal_eval)

    meta_list = celebrity_description.apply(
        lambda l: [get_description(celebrity) for celebrity in l]
    )
    face_id["ShortDescription"] = meta_list.apply(
        lambda meta_list: get_summary(meta_list, "description")
    )
    face_id["LongDescription"] = meta_list.apply(
        lambda meta_list: get_summary(meta_list, "detail")
    )

    face_id.to_csv(base / "face_id.csv", index=False)


def crop_face(img: Image, bounds):
    height, width = img.height * bounds["Height"], img.width * bounds["Width"]
    top, left = img.height * bounds["Top"], img.width * bounds["Left"]
    face_crop = img.crop((left, top, left + width, top + height))
    face_crop = np.array(face_crop)
    face_crop = rrdn.predict(face_crop)

    return face_crop


def extract_face(img_id, bounds):
    img_path = base / "img" / f"{img_id}.png"
    img = Image.open(img_path).convert("RGB")

    face_crops = [crop_face(img, bounding_box) for bounding_box in bounds]
    result = DeepFace.analyze(
        img_path=face_crops, actions=["race"], enforce_detection=False
    )

    return result


def extract_face_list(row, feature):
    img_id = str(row["id"]).zfill(5)
    file = base / "face_meta" / "celebrity" / f"{img_id}.json"

    if file.exists():
        return

    bounds = [face["Face"]["BoundingBox"] for face in row[feature]]
    result = extract_face(img_id, bounds)

    with open(file, "w") as f:
        json.dump(result, f)

    return result


def crop_face(img: Image, width, height, left, top):
    height, width = img.height * height, img.width * width
    top, left = img.height * top, img.width * left

    face_crop = img.crop((left, top, left + width, top + height))

    return np.array(face_crop)


def extract_face(models, result):
    img_id = str(result["id"]).zfill(5)
    img_path = base / "img" / f"{img_id}.png"
    img = Image.open(img_path).convert("RGB")

    race = DeepFace.analyze(
        img_path=crop_face(
            img,
            result["boundingbox_width"],
            result["boundingbox_height"],
            result["boundingbox_left"],
            result["boundingbox_top"],
        ),
        actions=["race"],
        models=models,
        enforce_detection=False,
    )

    return race


def extract_race_meta():
    models = {"race": DeepFace.build_model("Race")}
    face_id = pd.read_csv(base / "faces/general.csv")
    face_id["race"] = face_id.apply(lambda result: extract_face(models, result), axis=1)
    face_id.to_csv(base / "temp.csv")
