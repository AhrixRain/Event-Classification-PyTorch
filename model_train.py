import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

from plot import confidence_score, plots
from predict import prediction
from train import build_model, unified_train
from utils import classifier_dataloader_cropped

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="Execute different functions based on input parameters."
    )
    parser.add_argument(
        "function",
        type=str,
        help="The function to execute: unified_train, prediction, plots, raytune, or confidence_score",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize images to 224x224 (needed for MobileNet)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet",
        choices=["mobilenet", "resnet18", "resnet34", "resnet50"],
        help="Model to use: mobilenet, resnet18, resnet34, or resnet50",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    if args.function == "unified_train":
        unified_train(args)
    elif args.function == "prediction":
        prediction(args)
    elif args.function == "plots":
        plots(args)
    elif args.function == "raytune":
        raytune(args)
    elif args.function == "confidence_score":
        confidence_score(args)
    else:
        print("Invalid function name.")

def raytune(args):
    """Ray Tune hyperparameter optimization."""
    model_name = args.model
    base_batch_size = args.batch_size

    print(f"Starting Ray Tune for {model_name}")
    print(f"Base batch size: {base_batch_size}")

    def load_data(batch_size):
        train_loader, _ = classifier_dataloader_cropped(batch_size, True)
        _, validation_loader = classifier_dataloader_cropped(batch_size, True)
        return train_loader, validation_loader

    def objective(config):
        train_loader, validation_loader = load_data(config["batch_size"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_kwargs = {"width_mult": config.get("width_mult", 1.0)}
        if model_name != "mobilenet":
            model_kwargs = {}

        model = build_model(model_name, **model_kwargs)
        model.to(device)

        if model_name == "mobilenet":
            criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
            if config.get("optimizer", "adam") == "adamw":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config["lr"],
                    weight_decay=config.get("weight_decay", 0.0002),
                )
            else:
                optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=config["lr"],
                momentum=config["momentum"],
            )

        max_epochs = config.get("max_epochs", 20)

        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0

            for image0s, image1s, labels in train_loader:
                image0s = image0s.float().to(device)
                image1s = image1s.float().to(device)
                labels = torch.tensor([label.to(device) for label in labels], dtype=torch.int64).to(
                    device
                )

                output = model(image0s, image1s)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * image0s.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            model.eval()
            validation_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for image0s, image1s, labels in validation_loader:
                    image0s = image0s.float().to(device)
                    image1s = image1s.float().to(device)
                    labels = torch.tensor(
                        [label.to(device) for label in labels], dtype=torch.int64
                    ).to(device)

                    output = model(image0s, image1s)
                    loss = criterion(output, labels)

                    _, predicted = torch.max(output, 1)
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

                    validation_loss += loss.item() * image0s.size(0)

            val_loss = validation_loss / len(validation_loader.dataset)
            val_accuracy = correct_predictions / total_samples

            train.report(
                {
                    "mean_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "train_loss": epoch_loss,
                    "epoch": epoch,
                }
            )

    if model_name == "mobilenet":
        search_space = {
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([16, 32]),
            "width_mult": tune.choice([0.75, 1.0, 1.25]),
            "label_smoothing": tune.uniform(0.0, 0.2),
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "optimizer": tune.choice(["adam", "adamw"]),
            "max_epochs": tune.choice([15, 20, 25]),
        }
    else:
        search_space = {
            "lr": tune.loguniform(1e-7, 1e-3),
            "momentum": tune.uniform(0.1, 0.9),
            "batch_size": tune.choice([16, 32]),
            "max_epochs": tune.choice([15, 20, 25]),
        }

    algo = OptunaSearch()
    trainable_with_gpu = tune.with_resources(objective, {"gpu": 0.33})

    output_dir = f"/home/ziqinl12/Desktop/Event-Calss/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            num_samples=32,
            metric="mean_accuracy",
            mode="max",
            search_alg=algo,
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": 50},
            name=f"raytune_{model_name}",
            local_dir=output_dir,
        ),
        param_space=search_space,
    )

    print("Starting Ray Tune optimization...")
    results = tuner.fit()

    best_result = results.get_best_result()
    print(f"\n=== Ray Tune Results for {model_name} ===")
    print(f"Best config: {best_result.config}")
    print(f"Best accuracy: {best_result.metrics['mean_accuracy']:.4f}")
    print(f"Best validation loss: {best_result.metrics['val_loss']:.4f}")

    results_file = f"{output_dir}/raytune_results_{model_name}.txt"
    with open(results_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Ray Tune Results for {model_name}\n")
        file_obj.write("=" * 50 + "\n")
        file_obj.write(f"Best config: {best_result.config}\n")
        file_obj.write(f"Best accuracy: {best_result.metrics['mean_accuracy']:.4f}\n")
        file_obj.write(f"Best validation loss: {best_result.metrics['val_loss']:.4f}\n")
        file_obj.write(f"Number of trials: {len(results)}\n")

    print(f"Results saved to: {results_file}")

    return best_result.config


if __name__ == "__main__":
    main()

