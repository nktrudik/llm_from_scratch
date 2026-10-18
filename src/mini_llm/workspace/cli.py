"""CLI каталога версий: публикация, splits и управляемое обучение."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mini_llm.workspace.catalog import list_datasets, list_runs
from mini_llm.workspace.datasets import publish_dataset
from mini_llm.workspace.training import (
    ManagedTrainingRequest,
    TrainingParameters,
    prepare_version_splits,
    run_managed_training,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить только явно выбранный этап; список не загружает модели."""

    parser = argparse.ArgumentParser(description="Версии датасетов и запуски моделей")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("migrate")
    publish = commands.add_parser("publish")
    publish.add_argument("source", type=Path)
    publish.add_argument("--dataset", required=True)
    publish.add_argument("--version", required=True)
    publish.add_argument("--title", required=True)
    split = commands.add_parser("split")
    split.add_argument("--dataset", required=True)
    split.add_argument("--version", required=True)
    train = commands.add_parser("train")
    train.add_argument("--dataset", required=True)
    train.add_argument("--version", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--resume-run")
    train.add_argument("--checkpoint", default="last.pt")
    train.add_argument("--parameters", type=Path, help="JSON с TrainingParameters")
    args = parser.parse_args(argv)
    try:
        result: object
        if args.command == "migrate":
            from mini_llm.workspace.migration import migrate_workspace

            result = migrate_workspace()
        elif args.command == "list":
            result = {
                "datasets": [item.model_dump(mode="json") for item in list_datasets()],
                "runs": [item.model_dump(mode="json") for item in list_runs()],
            }
        elif args.command == "publish":
            result = publish_dataset(
                args.source, args.dataset, args.version, args.title
            ).model_dump()
        elif args.command == "split":
            result = prepare_version_splits(args.dataset, args.version)
        else:
            parameters = (
                TrainingParameters.model_validate_json(args.parameters.read_text(encoding="utf-8"))
                if args.parameters
                else TrainingParameters()
            )
            result = run_managed_training(
                ManagedTrainingRequest(
                    dataset_id=args.dataset,
                    dataset_version=args.version,
                    model_id=args.model,
                    parameters=parameters,
                    parent_run_id=args.resume_run,
                    checkpoint_name=args.checkpoint,
                ),
                "cli",
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Операция не завершена: {error}")
        return 1
    return 0
