"""ObjaverseXL loader adapted from Microsoft's TRELLIS dataset toolkit.
Upstream: https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/datasets/ObjaverseXL.py
Only the output_dir -> root/download_root argument contract is adapted for TRELLIS.2.
"""
import os
import argparse
import pandas as pd
import objaverse.xl as oxl

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='ObjaverseXL source (sketchfab or github)')
    parser.add_argument('--processes', type=int, default=8,
                        help='Concurrent Objaverse downloads')

def get_metadata(source='sketchfab', **kwargs):
    if source == 'sketchfab':
        uri = 'hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv'
    elif source == 'github':
        uri = 'hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv'
    else:
        raise ValueError(f'Invalid ObjaverseXL source: {source}')
    return pd.read_csv(uri)

def download(metadata, root, download_root=None, processes=8, **kwargs):
    output_dir = download_root or root
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    annotations = oxl.get_annotations(download_dir=os.path.join(output_dir, 'objaverse_annotations'))
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]
    file_paths = oxl.download_objects(
        annotations,
        download_dir=raw_dir,
        processes=processes,
        save_repo_format='zip',
    )
    downloaded = {}
    metadata = metadata.set_index('file_identifier')
    for file_identifier, path in file_paths.items():
        sha256 = metadata.loc[file_identifier, 'sha256']
        downloaded[sha256] = os.path.relpath(path, output_dir)
    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])

def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', no_file=False):
    """Per-instance dispatch adapted verbatim in behavior from Microsoft TRELLIS."""
    import tempfile
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    rows = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(rows), desc=desc) as pbar:
        def worker(row):
            sha256 = row['sha256']
            try:
                if no_file:
                    record = func(None, row)
                else:
                    local_path = row['local_path']
                    if local_path.startswith('raw/github/repos/'):
                        parts = local_path.split('/')
                        filename = os.path.join(*parts[5:])
                        zip_file = os.path.join(output_dir, *parts[:5])
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with zipfile.ZipFile(zip_file, 'r') as zf:
                                zf.extractall(tmp_dir)
                            record = func(os.path.join(tmp_dir, filename), row)
                    else:
                        record = func(os.path.join(output_dir, local_path), row)
                if record is not None:
                    records.append(record)
            except Exception as exc:
                print(f'Error processing object {sha256}: {exc}')
            finally:
                pbar.update()

        list(executor.map(worker, rows))
    return pd.DataFrame.from_records(records)
