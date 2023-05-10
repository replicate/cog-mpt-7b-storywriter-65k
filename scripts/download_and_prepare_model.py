#!/usr/bin/env python


import os
import shutil
import argparse 
import logging 
import sys 
import torch

from distutils.dir_util import copy_tree
from pathlib import Path
from tempfile import TemporaryDirectory
from huggingface_hub import snapshot_download, login
from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from tensorize_model import tensorize_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def download_model_from_hf_hub(
        model_name: str, 
        model_path: str,
        rm_existing_model: bool = True,
    ) -> dict:
    """
    This function downloads a model from the Hugging Face Hub and saves it locally.
    It also saves the tokenizer in a separate location so that it can be easely included in a docker Image
    without including the model weights.

    Args:
        model_name (str): Name of model on hugging face hub
        path (str): Local path where model is saved
        rm_existing_model (bool, optional): Whether to remove the existing model or not. Defaults to False.

    Returns:
        dict: Dictionary containing the model name and path
    """

    # model_weights_path = os.path.join(os.getcwd(), "model_weights/torch_weights")
    # model_path = os.path.join(model_weights_path, model_name)


    if rm_existing_model:
        logger.info(f"Removing existing model at {model_path}")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

    # setup temporary directory
    with TemporaryDirectory() as tmpdir:
        logger.info(f"Downloading {model_name} weights to temp...")

        snapshot_dir = snapshot_download(
            repo_id=model_name, 
            cache_dir=tmpdir,
            allow_patterns=["*.bin", "*.json", "*.md", "tokenizer.model", "checkpoint.pt", "*.py"],
        )
        # copy snapshot to model dir
        logger.info(f"Copying weights to {model_path}...")
        copy_tree(snapshot_dir, str(model_path))
    
    return {"model_name": model_name, "model_path": model_path}


def download_hf_model_and_copy_tokenizer(
        model_name: str,
        model_path: str,
        tokenizer_path: str,
        rm_existing_model: bool = True,
):    

    model_info = download_model_from_hf_hub(model_name, model_path)

    if tokenizer_path:
        # Move tokenizer to separate location
        logging.info(f"Copying tokenizer and model config to {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        tokenizer.save_pretrained(tokenizer_path)

        # Set the source and destination file paths
        config_path = os.path.join(model_path, "config.json")

        # Use the shutil.copy() function to copy the file to the destination directory
        shutil.copy(config_path, tokenizer_path)

    return model_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)


    parser.add_argument("--tensorize", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="fp32")

    args = parser.parse_args()
    if args.hf_token is not None:
        login(token=args.hf_token)

    download_hf_model_and_copy_tokenizer(args.model_name, model_path=args.model_path, tokenizer_path=args.tokenizer_path)

    if args.tensorize:
        model = tensorize_model(args.model_name, dtype=args.dtype, tensorizer_path=args.model_path)
