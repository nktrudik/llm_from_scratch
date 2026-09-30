"""Адаптеры между HTTP-схемами и существующими pipeline проекта."""

from __future__ import annotations

from dataclasses import asdict

from mini_llm.api.schemas import (
    DatasetSplitRequest,
    PreprocessingRequest,
    ScrapeRequest,
    TokenizerTrainingRequest,
    TokenStatisticsRequest,
    TrainingRequest,
)
from mini_llm.bpe_tokenizer import train_bpe_tokenizer
from mini_llm.dataset_split import SplitConfig, split_dataset
from mini_llm.preprocessing import preprocess_dataset
from mini_llm.scraper import TwoChScraper
from mini_llm.token_statistics import TokenStatisticsConfig, calculate_token_statistics
from mini_llm.training import train_model


def run_training(request: TrainingRequest) -> dict[str, object]:
    """Выполнить training pipeline и вернуть компактный итог."""

    result = train_model(request.to_config())
    return {
        "global_step": result.global_step,
        "best_validation_loss": result.best_validation_loss,
        "last_checkpoint": str(result.last_checkpoint),
        "interrupted": result.interrupted,
    }


def run_tokenizer_training(request: TokenizerTrainingRequest) -> dict[str, object]:
    """Обучить BPE tokenizer только на переданном train split."""

    tokenizer = train_bpe_tokenizer(
        request.train_file,
        request.output_file,
        vocab_size=request.vocab_size,
        min_frequency=request.min_frequency,
    )
    return {"tokenizer_file": str(request.output_file), "vocabulary_size": tokenizer.vocab_size}


def run_preprocessing(request: PreprocessingRequest) -> dict[str, object]:
    """Выполнить preprocessing и вернуть итоговую статистику."""

    return preprocess_dataset(request.to_config()).to_dict()


def run_dataset_split(request: DatasetSplitRequest) -> dict[str, object]:
    """Создать train/validation/test split и вернуть отчёт."""

    return split_dataset(SplitConfig(**request.model_dump()))


def run_token_statistics(request: TokenStatisticsRequest) -> dict[str, object]:
    """Посчитать token statistics и сохранить JSON-отчёт."""

    return calculate_token_statistics(TokenStatisticsConfig(**request.model_dump()))


def run_scraper(request: ScrapeRequest) -> dict[str, object]:
    """Последовательно собрать выбранные треды через JSON API."""

    config = request.to_config()
    with TwoChScraper(config) as scraper:
        paths = scraper.scrape_to_files(request.url, max_threads=request.max_threads)
    return {
        "saved_threads": len(paths),
        "files": [str(path) for path in paths],
        "config": asdict(config),
    }
