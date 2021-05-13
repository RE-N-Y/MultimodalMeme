from global_config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
from transformers import AutoTokenizer, AutoModel, ViTModel


class ConcatModel(nn.Module):
    def __init__(self, d_hidden=128, dropout=0.1, **kwargs):
        super().__init__()
        self.bow_celebrity_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, d_hidden),
            nn.ReLU(),
        )
        self.bow_caption_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, d_hidden),
            nn.ReLU(),
        )
        self.img_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, d_hidden),
            nn.ReLU(),
        )
        self.text_fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(768, d_hidden), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4 * d_hidden, 2 * d_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, img, text, bow_celebrity, bow_caption, **kwargs):
        celebrity_logit = self.bow_celebrity_fc(bow_celebrity)
        caption_logit = self.bow_caption_fc(bow_caption)
        img_logit, text_logit = self.img_fc(img), self.text_fc(text)
        logits = torch.cat(
            (img_logit, text_logit, celebrity_logit, caption_logit), dim=-1
        )
        logits = self.classifier(logits)

        return logits
