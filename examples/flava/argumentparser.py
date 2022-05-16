import argparse
import os


def get_training_arg_parser():
    parser = argparse.ArgumentParser(description='Training FLAVA model')
    # Datasets
    parser.add_argument('--imagenet_train_root', type=str, default=os.getenv('IMAGENET_TRAIN_ROOT', None))
    parser.add_argument('--imagenet_val_root', type=str, default=os.getenv('IMAGENET_VAL_ROOT', None))
    parser.add_argument('--save_dir', type=str, default=os.getenv('SAVE_DIR', None))
    parser.add_argument('--hf_dir', type=str, default=os.getenv('HF_DIR', None))
    parser.add_argument('--pyt_dir', type=str, default=os.getenv('PYT_DIR', None))
    parser.add_argument('--num_proc', type=int, default=32, help='number of threads for downloading data')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--allow_uneven_batches', nargs='?', default=False, const=True)
    # Lighting
    parser.add_argument('--gpus', type=int, default=-1, help='number of gpus to run')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--max_steps', type=int, default=450000)
    parser.add_argument('--parallel_strategy', type=str, default='ddp')
    parser.add_argument('--sanity_steps', type=int, default=0)
    return parser
