# Run the full pipeline from device selection through ONNX export.

from no_emoji import strip_emojis_streams
strip_emojis_streams()  # Filter library output before importing it.

from device import get_device
from trainer import train_model
from evaluator import load_best_model, validate_model
from exporter import export_model


def main():
    device = get_device()
    print(f"Using: {device}\n")

    model, device = train_model(device)

    trained = load_best_model(fallback_model=model)
    validate_model(trained, device)
    export_model(trained, fmt="onnx")


if __name__ == "__main__":
    main()
