import os
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train import build_model
from utils import classifier_dataloader_cropped


def prediction(args) -> Tuple[str, float]:
    """Run prediction using a trained model and persist results."""
    model_name = args.model
    batch_size = args.batch_size

    model_dir = f"./{model_name}"
    model_path = f"{model_dir}/{model_name}_best.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(
            f"Please train the model first using: python model_train.py unified_train --model {model_name}"
        )
        return "", 0.0

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    print(f"Running prediction for {model_name}")
    print(f"Model path: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Resize images: {args.resize}")

    _, test_loader = classifier_dataloader_cropped(batch_size, shuffle=False)

    model = build_model(model_name)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Model validation accuracy: {checkpoint.get('accuracy', 0.0):.2f}%")
        else:
            model = checkpoint

        model.to(device)
        model.eval()
        print("Model loaded successfully!")

    except Exception as err:  # pylint: disable=broad-except
        print(f"Error loading model: {err}")
        return "", 0.0

    if device.type == "cuda":
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

    probs_list = []
    labels_list = []
    files_list = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (img0, img1, labels) in enumerate(tqdm(test_loader, desc="Inference")):
            paths = [f"sample_{batch_idx * batch_size + i}" for i in range(len(labels))]

            img0 = img0.float().to(device)
            img1 = img1.float().to(device)
            labels = labels.to(device)

            logits = model(img0, img1)
            probs = F.softmax(logits, dim=1)

            probs_list.append(probs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            files_list.extend([p.encode("utf-8") for p in paths])

    probs_arr = np.concatenate(probs_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)

    prediction_dir = "./prediction"
    os.makedirs(prediction_dir, exist_ok=True)

    out_name = f"pred_{model_name}.h5"
    out_path = os.path.join(prediction_dir, out_name)

    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("probs", data=probs_arr)
        hf.create_dataset("labels", data=labels_arr.astype(np.int32))
        dt = h5py.special_dtype(vlen=bytes)
        hf.create_dataset("files", data=np.array(files_list, dtype=dt))

    print(f"Prediction results saved to: {out_path}")
    print(f"Probabilities shape: {probs_arr.shape}")
    print(f"Labels shape: {labels_arr.shape}")

    preds = np.argmax(probs_arr, axis=1)
    accuracy = np.mean(preds == labels_arr)
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    class_names = ["nuecc", "numucc", "nc"]
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = labels_arr == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(preds[class_mask] == labels_arr[class_mask])
            print(
                f"  {class_name}: {class_acc:.4f} ({class_acc * 100:.2f}%) - {np.sum(class_mask)} samples"
            )
        else:
            print(f"  {class_name}: No samples found")

    stats_file = os.path.join(prediction_dir, f"pred_{model_name}_stats.txt")
    with open(stats_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Prediction Results for {model_name}\n")
        file_obj.write("=" * 50 + "\n")
        file_obj.write(f"Model path: {model_path}\n")
        file_obj.write(f"Output file: {out_path}\n")
        file_obj.write(f"Total samples: {len(labels_arr)}\n")
        file_obj.write(f"Overall accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        file_obj.write(f"Probabilities shape: {probs_arr.shape}\n")
        file_obj.write(f"Labels shape: {labels_arr.shape}\n")
        file_obj.write("\nPer-class accuracy:\n")
        for i, class_name in enumerate(class_names):
            class_mask = labels_arr == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(preds[class_mask] == labels_arr[class_mask])
                file_obj.write(
                    f"  {class_name}: {class_acc:.4f} ({class_acc * 100:.2f}%) - {np.sum(class_mask)} samples\n"
                )
            else:
                file_obj.write(f"  {class_name}: No samples found\n")

        file_obj.write("\nClass distribution:\n")
        for i, class_name in enumerate(class_names):
            count = np.sum(labels_arr == i)
            file_obj.write(f"  {class_name}: {count} samples ({100 * count / len(labels_arr):.1f}%)\n")

    print(f"Prediction statistics saved to: {stats_file}")

    return out_path, accuracy

