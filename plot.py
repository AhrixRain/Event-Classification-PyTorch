import os
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

from train import build_model
from utils import (
    classifier_dataloader_cropped,
    class_pid_plots,
    infer_class_labels,
    load_predictions,
    plot_calculated_matrices,
    plot_confusion_matrix,
    sample_distribution_plot,
)


def plots(args) -> Tuple[float, float, Dict[str, Any]]:
    """Generate analysis plots using stored predictions."""
    model_name = args.model

    prediction_file = f"./prediction/pred_{model_name}.h5"
    output_dir = f"./{model_name}"

    if not os.path.exists(prediction_file):
        print(f"Error: Prediction file not found at {prediction_file}")
        print(f"Please run prediction first using: python model_train.py prediction --model {model_name}")
        return 0.0, 0.0, {}

    print(f"Generating plots for {model_name}")
    print(f"Prediction file: {prediction_file}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        probs, labels, files = load_predictions(prediction_file)
        n_classes = probs.shape[1]
        class_names = infer_class_labels(n_classes)

        print(f"Loaded predictions: {probs.shape}, classes: {n_classes}")
        print(f"Loaded labels shape: {labels.shape}")
        print(f"Using class names: {class_names}")

    except Exception as err:  # pylint: disable=broad-except
        print(f"Error loading prediction file: {err}")
        return 0.0, 0.0, {}

    print("Label distribution:")
    for i in range(n_classes):
        count = np.sum(labels == i)
        percent = count / len(labels) * 100
        print(f"  {class_names[i]}: {count} ({percent:.1f}%)")
        scores = probs[:, i]
        print(
            f"  {class_names[i]} prediction scores: mean={scores.mean():.4f}, "
            f"max={scores.max():.4f}, min={scores.min():.4f}"
        )

    pred = np.argmax(probs, axis=1)

    class_counts = [0, 0, 0]
    for prob_vec in probs:
        class_idx = np.argmax(prob_vec)
        class_counts[class_idx] += 1
    print(
        f"Predicted class distribution: NueCC: {class_counts[0]}, "
        f"NumuCC: {class_counts[1]}, NC: {class_counts[2]}"
    )

    conf = confusion_matrix(labels, pred, labels=range(n_classes))
    f1 = f1_score(labels, pred, labels=range(n_classes), average="weighted")
    print(f"F1 Harmonic: {f1:.4f}")
    print("Generating plots...")

    sample_distribution_plot(labels, pred, class_names, output_dir, model_name)
    print("  ✓ Sample distribution plot saved")

    metrics = class_pid_plots(probs, labels, class_names, output_dir, model_name)
    print("  ✓ Class PID analysis plots saved")

    plot_confusion_matrix(
        conf.astype(float),
        class_names,
        os.path.join(output_dir, f"confusion_counts_{model_name}.png"),
        "Confusion Matrix (Event Counts)",
    )
    print("  ✓ Confusion matrix (counts) saved")

    row_norm = conf.astype(np.float32) / np.sum(conf, axis=1)[:, np.newaxis]
    plot_confusion_matrix(
        row_norm,
        class_names,
        os.path.join(output_dir, f"confusion_eff_{model_name}.png"),
        "Efficiency (Row Normalized)",
    )
    print("  ✓ Efficiency confusion matrix saved")

    col_norm = conf.astype(np.float32) / np.sum(conf, axis=0)
    plot_confusion_matrix(
        col_norm,
        class_names,
        os.path.join(output_dir, f"confusion_pur_{model_name}.png"),
        "Purity (Column Normalized)",
    )
    print("  ✓ Purity confusion matrix saved")

    plot_calculated_matrices(class_names, output_dir, model_name, labels, pred)
    print("  ✓ Calculated matrices saved")

    summary_path = os.path.join(output_dir, f"metrics_summary_{model_name}.txt")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Analysis Results for {model_name}\n")
        file_obj.write("=" * 50 + "\n")
        file_obj.write(f"Prediction file: {prediction_file}\n")
        file_obj.write(f"Total samples: {len(labels)}\n")
        file_obj.write(f"Number of classes: {n_classes}\n")
        file_obj.write(f"Class names: {class_names}\n")
        file_obj.write(f"Overall F1 score: {f1:.4f}\n\n")

        for key, value in metrics.items():
            file_obj.write(
                f"{key}: threshold={value['threshold']:.4f} eff={value['eff']:.4f} "
                f"pur={value['pur']:.4f} fom={value['fom']:.4f} auc={value['auc']:.4f} "
                f"ap={value['avg_precision']:.4f}\n"
            )

            file_obj.write("  Selections:\n")
            for cls, sel in value["class_sel"].items():
                file_obj.write(f"    {cls}: {sel}\n")

            file_obj.write("  Efficiencies:\n")
            for cls, eff in value["class_eff"].items():
                file_obj.write(f"    {cls}: {eff:.4f}\n")

            file_obj.write("  Purities:\n")
            for cls, pur in value["class_pur"].items():
                file_obj.write(f"    {cls}: {pur:.4f}\n")

            file_obj.write("\n")

        file_obj.write("Label distribution:\n")
        for i, class_name in enumerate(class_names):
            count = np.sum(labels == i)
            percent = count / len(labels) * 100
            file_obj.write(f"  {class_name}: {count} samples ({percent:.1f}%)\n")

        file_obj.write("\nPrediction distribution:\n")
        for i, class_name in enumerate(class_names):
            count = np.sum(pred == i)
            percent = count / len(pred) * 100
            file_obj.write(f"  {class_name}: {count} samples ({percent:.1f}%)\n")

    print(f"  ✓ Metrics summary saved to: {summary_path}")

    accuracy = np.mean(pred == labels)
    print(f"\n=== Analysis Summary ===")
    print(f"Model: {model_name}")
    print(f"Total samples: {len(labels)}")
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"F1 score: {f1:.4f}")
    print(f"All plots saved to: {output_dir}")

    return accuracy, f1, metrics


def confidence_score(args) -> None:
    """Analyze model confidence scores on the validation split."""
    model_name = args.model
    batch_size = args.batch_size

    output_dir = f"./{model_name}"
    model_path = f"{output_dir}/{model_name}_best.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please train the model first using: python model_train.py unified_train --model {model_name}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {model_name} on: {str(device)}")
    print(f"Model path: {model_path}")

    _, validation_loader = classifier_dataloader_cropped(batch_size, True)

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
        return

    probabilities = []
    true_labels = []
    total_samples = 0
    correct_predictions = 0

    print("Collecting confidence scores...")
    with torch.no_grad():
        for image0s, image1s, labels in tqdm(validation_loader, desc="Processing"):
            image0s = image0s.float().to(device)
            image1s = image1s.float().to(device)
            labels = torch.tensor([label.to(device) for label in labels], dtype=torch.int64).to(
                device
            )

            output = model(image0s, image1s)

            probs = F.softmax(output, dim=1)
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            _, predicted = torch.max(output, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)

    accuracy = 100.0 * correct_predictions / total_samples
    print("Evaluation Results:")
    print(f"Total samples: {total_samples}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Probabilities shape: {probabilities.shape}")

    sherpa_output_dir = f"/home/houyh/neutrino-cnn/{model_name}"
    os.makedirs(sherpa_output_dir, exist_ok=True)

    class_names = ["nuecc", "numucc", "nc"]
    colors = ["blue", "orange", "green"]

    for i in range(3):
        if np.sum(true_labels == i) == 0:
            print(f"Warning: No samples found for class {class_names[i]}")
            continue

        plt.figure(figsize=(10, 6))

        for j in range(3):
            class_probs = probabilities[true_labels == i, j]
            if len(class_probs) > 0:
                sns.kdeplot(
                    class_probs,
                    label=f"{class_names[j]} Probability",
                    fill=True,
                    alpha=0.6,
                    color=colors[j],
                )

        plt.title(f"Confidence Score Distributions for True {class_names[i]} (n={np.sum(true_labels == i)})")
        plt.xlabel("Confidence Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)

        plt.savefig(
            f"{sherpa_output_dir}/confidence_scores_{class_names[i]}_kde.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved confidence plot for {class_names[i]}")

    print(f"All confidence score plots saved to: {sherpa_output_dir}")

    summary_file = f"{sherpa_output_dir}/confidence_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Model: {model_name}\n")
        file_obj.write(f"Model path: {model_path}\n")
        file_obj.write(f"Total samples: {total_samples}\n")
        file_obj.write(f"Accuracy: {accuracy:.2f}%\n")
        file_obj.write("Class distribution:\n")
        for i, class_name in enumerate(class_names):
            count = np.sum(true_labels == i)
            file_obj.write(f"  {class_name}: {count} samples ({100 * count / total_samples:.1f}%)\n")

    print(f"Summary saved to: {summary_file}")

