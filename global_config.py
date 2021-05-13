import os, re
import torch
import numpy as np
import subprocess
import random


def get_free_gpu():
    command = "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free"
    lines = subprocess.check_output(command, shell=True)
    memory = []
    for line in lines.splitlines():
        memory.extend(re.findall(r"\d+", str(line)))
    memory = [int(m) for m in memory]

    return str(np.argmax(memory))


def set_random_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda")
D_CONTEXT = 1024
D_VISUAL = 2048
D_TEXT = 768
