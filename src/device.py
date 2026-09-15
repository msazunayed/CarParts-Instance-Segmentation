#Figures out whether we can actually use the GPU or should just stick to CPU."""

import torch
import config


def get_device() -> str:
    #Returns '0' if there's a usable GPU with enough free memory, else 'cpu'
    if not torch.cuda.is_available():
        print("No GPU found!")
        return "cpu"

    try:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        print(f"GPU detected: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

        # Avoid starting a run that will run out of memory immediately.
        if free_gb < config.MIN_FREE_GB:
            print("Very little free GPU memory - falling back to CPU")
            return "cpu"

        return "0"

    except RuntimeError as e:
        # Use the CPU if the driver cannot initialize the GPU.
        print(f"GPU existed but could not be initialized ({e}). Using CPU instead.")
        return "cpu"
