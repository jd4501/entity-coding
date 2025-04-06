import argparse
import os
import zipfile
import tarfile
import requests
import gdown
import yaml
from tqdm import tqdm

def download_from_drive(url, output_path):
    """
    Download a file from Google Drive using gdown.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(url, output_path, quiet=False, fuzzy=True)

def download_from_url(url, output_path):
    """
    Download a file from an external URL with a progress bar.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    with open(output_path, 'wb') as file, tqdm(
        desc=f"📦 Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

def unzip_file(zip_path, extract_to):
    """
    Extract a ZIP archive.
    """
    print(f"\n🗜️ Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.infolist()
        os.makedirs(extract_to, exist_ok=True)
        for file in tqdm(file_list, desc="Extracting", unit="file"):
            zip_ref.extract(file, extract_to)
    print("✅ Unzipping complete.")

def untar_file(tar_path, extract_to):
    """
    Extract a TAR.GZ archive.
    """
    print(f"\n🗜️ Extracting TAR.GZ {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        os.makedirs(extract_to, exist_ok=True)
        for member in tqdm(members, desc="Extracting", unit="file"):
            tar.extract(member, extract_to)
    print("✅ Extraction complete.")

def process_download(url, target_dir, cleanup, is_external=False):
    """
    Download and extract a file from a given URL into the target directory.

    Uses gdown for Google Drive URLs or requests for external URLs, and
    extracts based on file extension (.zip or .tar.gz). If the file is not an
    archive, it is moved directly.
    """
    os.makedirs(target_dir, exist_ok=True)
    temp_dir = "downloads"
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.basename(url)
    archive_path = os.path.join(temp_dir, filename)
    
    print(f"\n📥 Downloading file from {url}...")
    if is_external:
        download_from_url(url, archive_path)
    else:
        download_from_drive(url, archive_path)
    
    # Extract based on file extension or simply move the file if not an archive.
    if archive_path.endswith(".zip"):
        unzip_file(archive_path, target_dir)
    elif archive_path.endswith(".tar.gz"):
        untar_file(archive_path, target_dir)
    else:
        target_file = os.path.join(target_dir, filename)
        os.replace(archive_path, target_file)
        print(f"Downloaded file saved to {target_file}")
    
    if cleanup and os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"🧹 Deleted archive: {archive_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract model files using URLs from a YAML config file."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/download_config.yaml',
        help='Path to the YAML configuration file (default: config/download_config.yaml)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Delete archive files after extraction'
    )
    args = parser.parse_args()

    # Load configuration from YAML file.
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    # Process NER Model (Google Drive)
    if config_data.get('ner'):
        target = os.path.join('data', 'models', 'ner_model')
        process_download(config_data['ner'], target, args.cleanup, is_external=False)

    # Process AC Model (Google Drive)
    if config_data.get('ac'):
        target = os.path.join('data', 'models', 'ac_model')
        process_download(config_data['ac'], target, args.cleanup, is_external=False)

    # Process RoBERTa Model (External URL)
    if config_data.get('roberta'):
        target = os.path.join('data', 'models', 'RoBERTa-base-PM-M3-Voc-distill-align-hf')
        process_download(config_data['roberta'], target, args.cleanup, is_external=True)

    # Process Entity-only Model and Tokenizer (Google Drive)
    if config_data.get('entity'):
        entity_config = config_data['entity']
        if 'model' in entity_config:
            target_entity = os.path.join('modules', 'plm-ca', 'models', 'entityonly')
            process_download(entity_config['model'], target_entity, args.cleanup, is_external=False)
        if 'tokenizer' in entity_config:
            target_tokenizer = os.path.join('modules', 'plm-ca', 'models', 'tokenizer_latest')
            process_download(entity_config['tokenizer'], target_tokenizer, args.cleanup, is_external=False)

    # Process Full-text Model (Google Drive)
    if config_data.get('fulltext'):
        target = os.path.join('modules', 'plm-ca', 'models', 'fulltext')
        process_download(config_data['fulltext'], target, args.cleanup, is_external=False)

if __name__ == "__main__":
    main()
