"""Download the model files used by the entity-coding pipeline.

This script reads `config/download_config.yaml`, downloads the selected model
archives, and extracts them into the repo-owned locations used by the NER,
assertion classification, and PLM-CA inference scripts. By default it downloads
all released model artefacts. Use `--models` to fetch only part of the bundle
and `--cleanup` to remove downloaded archives after extraction.
"""

import argparse
import logging
import re
import string
import tarfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import gdown
import requests
import yaml
from tqdm import tqdm

LOGGER = logging.getLogger("data_download")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "download_config.yaml"
DOWNLOAD_DIR = REPO_ROOT / "downloads"

CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = 30
MODEL_CHOICES = ("all", "ner", "ac", "roberta", "entity-only", "fulltext")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class DownloadTask:
    label: str
    url: str
    target_dir: Path
    filename: str
    is_external: bool = False


class ConfigError(ValueError):
    """Raised when the download config is missing a selected model URL."""


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )


def display_path(path: Path) -> str:
    """Return a compact repo-relative path when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve CLI paths relative to the repo root, matching top-level scripts."""
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def sanitize_filename(name: str) -> str:
    """Return a filename that is portable across Windows and POSIX filesystems."""
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    return "".join(c for c in name if c in valid_chars) or "downloaded_file"


def google_drive_file_id(url: str) -> str:
    """Extract a file ID from common Google Drive file URL formats."""
    parsed = urlparse(url)
    query_id = parse_qs(parsed.query).get("id", [None])[0]
    if query_id:
        return query_id

    match = re.search(r"/d/([a-zA-Z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)

    raise ValueError(
        "Invalid Google Drive file URL. Expected a URL containing '/d/<file_id>' "
        "or an 'id=<file_id>' query parameter."
    )


def finalize_download(part_path: Path, output_path: Path, source: str) -> Path:
    if not part_path.exists() or part_path.stat().st_size == 0:
        raise RuntimeError(f"Download from {source} produced an empty file at {output_path}.")
    part_path.replace(output_path)
    return output_path


def download_from_drive(url: str, output_path: Path) -> Path:
    """Download a Google Drive file to a temporary `.part` file before publishing it."""
    file_id = google_drive_file_id(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_output_path = output_path.with_name(sanitize_filename(output_path.name))
    part_path = clean_output_path.parent / f"{clean_output_path.name}.part"
    part_path.unlink(missing_ok=True)

    LOGGER.info("Downloading Google Drive file to %s", display_path(clean_output_path))
    try:
        result = gdown.download(url=url, output=str(part_path), quiet=True, fuzzy=True)
        if not result:
            raise RuntimeError(
                f"gdown failed to download Google Drive file id={file_id} to {clean_output_path}. "
                "The file may be private, deleted, or rate-limited; try again or check the URL."
            )
        return finalize_download(part_path, clean_output_path, f"Google Drive file id={file_id}")
    finally:
        part_path.unlink(missing_ok=True)


def download_from_url(url: str, output_path: Path) -> Path:
    """Download an external URL to a temporary `.part` file before publishing it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.parent / f"{output_path.name}.part"
    part_path.unlink(missing_ok=True)

    LOGGER.info("Downloading URL to %s", display_path(output_path))
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with part_path.open("wb") as file, tqdm(
                desc=f"Downloading {output_path.name}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not LOGGER.isEnabledFor(logging.INFO),
            ) as bar:
                for data in response.iter_content(CHUNK_SIZE):
                    if not data:
                        continue
                    file.write(data)
                    bar.update(len(data))
        return finalize_download(part_path, output_path, url)
    finally:
        part_path.unlink(missing_ok=True)


def _ensure_within_directory(target_dir: Path, member_path: str) -> None:
    """Reject archive members that would extract outside the target directory."""
    target_real = target_dir.resolve()
    member_real = (target_dir / member_path).resolve()
    try:
        member_real.relative_to(target_real)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to extract archive member outside target directory: {member_path!r}"
        ) from exc


def unzip_file(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP archive, rejecting members that would escape the target directory."""
    LOGGER.info("Unzipping %s to %s", display_path(zip_path), display_path(extract_to))
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.infolist()
        extract_to.mkdir(parents=True, exist_ok=True)
        for file in tqdm(
            file_list,
            desc="Extracting",
            unit="file",
            disable=not LOGGER.isEnabledFor(logging.INFO),
        ):
            _ensure_within_directory(extract_to, file.filename)
            zip_ref.extract(file, extract_to)
    LOGGER.info("Unzipping complete.")


def untar_file(tar_path: Path, extract_to: Path) -> None:
    """Extract a TAR.GZ archive, rejecting members that would escape the target directory."""
    LOGGER.info("Extracting TAR.GZ %s to %s", display_path(tar_path), display_path(extract_to))
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        extract_to.mkdir(parents=True, exist_ok=True)
        for member in tqdm(
            members,
            desc="Extracting",
            unit="file",
            disable=not LOGGER.isEnabledFor(logging.INFO),
        ):
            _ensure_within_directory(extract_to, member.name)
            if member.islnk() or member.issym():
                raise RuntimeError(
                    f"Refusing to extract link member from tar archive: {member.name!r}"
                )
            tar.extract(member, extract_to)
    LOGGER.info("Extraction complete.")


def process_download(task: DownloadTask, cleanup: bool) -> None:
    """Download one configured artefact and extract it into its repo-owned directory."""
    task.target_dir.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = task.filename
    if task.is_external:
        filename = sanitize_filename(Path(urlparse(task.url).path).name)
    archive_path = DOWNLOAD_DIR / filename

    LOGGER.info("Processing %s", task.label)
    if task.is_external:
        archive_path = download_from_url(task.url, archive_path)
    else:
        archive_path = download_from_drive(task.url, archive_path)

    if archive_path.name.endswith(".zip"):
        unzip_file(archive_path, task.target_dir)
    elif archive_path.name.endswith(".tar.gz"):
        untar_file(archive_path, task.target_dir)
    else:
        target_file = task.target_dir / filename
        archive_path.replace(target_file)
        LOGGER.info("Downloaded file saved to %s", display_path(target_file))

    if cleanup and archive_path.exists():
        archive_path.unlink()
        LOGGER.info("Deleted archive: %s", display_path(archive_path))


def parse_models_arg(value: str) -> set[str]:
    """Parse the comma-separated --models value into a set of selected model names."""
    selected = {item.strip() for item in value.split(",") if item.strip()}
    invalid = selected - set(MODEL_CHOICES)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown model selector(s): {sorted(invalid)}. Choose from: {MODEL_CHOICES}."
        )
    if not selected:
        raise argparse.ArgumentTypeError("Select at least one model.")
    if "all" in selected:
        return {"ner", "ac", "roberta", "entity-only", "fulltext"}
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract model files using URLs from a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file (default: config/download_config.yaml).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete archive files after extraction.",
    )
    parser.add_argument(
        "--models",
        type=parse_models_arg,
        default=parse_models_arg("all"),
        help=(
            "Comma-separated list of models to download. "
            "Choices: all (default - every model), ner (NER model), ac (AC model), "
            "roberta (RoBERTa-PM encoder; init weights for retraining), "
            "entity-only (entity-only ICD-10 coding model + matching tokenizer), "
            "fulltext (full-text ICD-10 coding model). "
            "Example: --models ner,ac,entity-only"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Mapping[str, object]:
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {display_path(config_path)}")

    with config_path.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file) or {}
    if not isinstance(config_data, Mapping):
        raise ConfigError(
            f"Config file must contain a YAML mapping: {display_path(config_path)}"
        )
    return config_data


def required_url(config_data: Mapping[str, object], key_path: tuple[str, ...]) -> str:
    current: object = config_data
    for key in key_path[:-1]:
        if not isinstance(current, Mapping) or not isinstance(current.get(key), Mapping):
            raise ConfigError(f"Missing config section: {'.'.join(key_path[:-1])}")
        current = current[key]

    final_key = key_path[-1]
    if not isinstance(current, Mapping):
        raise ConfigError(f"Missing config section: {'.'.join(key_path[:-1])}")
    value = current.get(final_key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing URL in config: {'.'.join(key_path)}")
    return value


def build_download_plan(config_data: Mapping[str, object], selected: set[str]) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []

    if "ner" in selected:
        tasks.append(
            DownloadTask(
                label="NER model",
                url=required_url(config_data, ("ner",)),
                target_dir=REPO_ROOT / "data" / "models",
                filename="ner_model.zip",
            )
        )

    if "ac" in selected:
        tasks.append(
            DownloadTask(
                label="assertion classification model",
                url=required_url(config_data, ("ac",)),
                target_dir=REPO_ROOT / "data" / "models",
                filename="ac_model.zip",
            )
        )

    if "roberta" in selected:
        tasks.append(
            DownloadTask(
                label="RoBERTa-PM encoder",
                url=required_url(config_data, ("roberta",)),
                target_dir=(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "RoBERTa-base-PM-M3-Voc-distill-align-hf"
                ),
                filename="roberta",
                is_external=True,
            )
        )

    if "entity-only" in selected:
        tasks.extend(
            [
                DownloadTask(
                    label="entity-only ICD-10 coding model",
                    url=required_url(config_data, ("entity", "model")),
                    target_dir=REPO_ROOT / "external" / "plm_ca" / "models",
                    filename="entityonly.zip",
                ),
                DownloadTask(
                    label="entity-aware tokenizer",
                    url=required_url(config_data, ("entity", "tokenizer")),
                    target_dir=REPO_ROOT / "external" / "plm_ca" / "models",
                    filename="tokenizer_latest.zip",
                ),
            ]
        )

    if "fulltext" in selected:
        tasks.append(
            DownloadTask(
                label="full-text ICD-10 coding model",
                url=required_url(config_data, ("fulltext",)),
                target_dir=REPO_ROOT / "external" / "plm_ca" / "models",
                filename="fulltext.zip",
            )
        )

    return tasks


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    config_path = resolve_repo_path(args.config)
    try:
        config_data = load_config(config_path)
        tasks = build_download_plan(config_data, args.models)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Selected models: %s", sorted(args.models))
    for task in tasks:
        process_download(task, args.cleanup)


if __name__ == "__main__":
    main()
