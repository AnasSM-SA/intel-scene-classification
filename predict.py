import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(description="Classify a natural-scene image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/intel_scene_cnn.keras"))
    parser.add_argument("--labels", type=Path, default=Path("models/class_names.json"))
    args = parser.parse_args()
    for path in (args.image, args.model, args.labels):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    model = keras.models.load_model(args.model)
    labels = json.loads(args.labels.read_text())
    size = tuple(model.input_shape[1:3])
    image = keras.utils.load_img(args.image, target_size=size, interpolation="bilinear")
    pixels = keras.utils.img_to_array(image)
    # Rescaling is already included in the trained model.
    scores = model.predict(pixels[None, ...], verbose=0)[0]
    if len(labels) != len(scores):
        raise ValueError("Class labels do not match the model output.")
    for index in np.argsort(scores)[::-1]:
        print(f"{labels[index]:12s} {scores[index]:.2%}")


if __name__ == "__main__":
    main()
