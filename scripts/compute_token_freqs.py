import os
import sys

import hydra
import numpy as np
import omegaconf
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import dataloader
import utils


def _iter_input_ids(dataset):
  for item in dataset:
    input_ids = item['input_ids']
    if not torch.is_tensor(input_ids):
      input_ids = torch.tensor(input_ids, dtype=torch.int64)
    yield input_ids


@hydra.main(version_base=None, config_path="../configs",
            config_name="config.yaml")
def main(config: omegaconf.DictConfig):
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  vocab_size = len(tokenizer)

  dataset = dataloader.get_dataset(
    config.data.train,
    tokenizer,
    mode='train',
    wrap=config.data.wrap,
    insert_eos=config.data.insert_train_eos,
    cache_dir=config.data.cache_dir,
    block_size=config.model.length,
    streaming=config.data.streaming,
    num_proc=config.loader.num_workers,
    revision=config.data.get("train_revision", None),
    custom_train_file=config.data.get("train_file", None),
    custom_valid_file=config.data.get("valid_file", None))

  counts = torch.zeros(vocab_size, dtype=torch.float64)
  total = 0
  for input_ids in _iter_input_ids(dataset):
    flat = input_ids.view(-1).to(torch.int64)
    counts += torch.bincount(flat, minlength=vocab_size).to(
      counts.dtype)
    total += flat.numel()

  if total == 0:
    raise ValueError('Dataset appears empty; no tokens counted.')

  freqs = (counts / total).clamp(min=1e-12)

  output_path = config.algo.get('token_freqs_path', None)
  if output_path is None:
    output_path = 'token_freqs.npy'
    logger.info(
      'algo.token_freqs_path not set; saving to %s', output_path)

  if not os.path.isabs(output_path):
    output_path = os.path.join(
      hydra.utils.get_original_cwd(), output_path)

  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  np.save(output_path, freqs.cpu().numpy())
  logger.info('Saved token frequencies to %s', output_path)


if __name__ == "__main__":
  main()
