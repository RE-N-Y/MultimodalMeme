import re
import numpy as np
import subprocess


def get_free_gpu():
    command = "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free"
    lines = subprocess.check_output(command, shell=True)
    memory = []
    for line in lines.splitlines():
        memory.extend(re.findall(r'\d+', str(line)))
    memory = [int(m) for m in memory]

    return str(np.argmax(memory))


if __name__ == "__main__":
    print(get_free_gpu())
