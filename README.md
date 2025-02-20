# TorchMultimodal (Alpha Release)

## Introduction
TorchMultimodal is a PyTorch library for training state-of-the-art multimodal multi-task models at scale. It provides:
- A repository of modular and composable building blocks (models, fusion layers, loss functions, datasets and utilities).
- A repository of examples that show how to combine these building blocks with components and common infrastructure from across the PyTorch Ecosystem to replicate state-of-the-art models published in the literature. These examples should serve as baselines for ongoing research in the field, as well as a starting point for future work.

As a first open source example, researchers will be able to train and extend FLAVA using TorchMultimodal.

## Running from the Docker container
The easiest way to use the torchmultimodal library is through the Docker container. To do so, first build the image with:

docker build --network=host -t multimodal:training .

And, run the image using:

 docker run --gpus all 
               -v datasets_dir/:datasets_dir \
               -v checkpoint_dir:checkpoint_dir \
               -v codebase:codebase \
              -e TRANSFORMERS_CACHE=transformers_cache_dir \
              -e IMAGENET_TRAIN_ROOT=imagenet_train_dir \
              -e IMAGENET_VAL_ROOT=imagenet_val_dir \
              -e HF_DIR=huggingface_datasets_cache_dir \
              -e PYT_DIR=pytorch_hub_cachedir \
              -e SAVE_DIR=checkpoint_save_dir \
             --network=host -ti --ipc=host multimodal:training 

python train.py

## Installation

TorchMultimodal requires Python >= 3.8. The library can be installed with or without CUDA support.

### Building from Source

1. Create conda environment
    ```
    conda create -n torch-multimodal python=<python_version>
    conda activate torch-multimodal
    ```
2. Install pytorch, torchvision, and torchtext. See [PyTorch documentation](https://pytorch.org/get-started/locally/).
   For now we only support Linux platform.
    ```
    conda install pytorch torchvision torchtext cudatoolkit=11.3 -c pytorch-nightly -c nvidia

    # For CPU-only install
    conda install pytorch torchvision torchtext cpuonly -c pytorch-nightly
    ```
3. Download and install TorchMultimodal and remaining requirements.
    ```
    git clone --recursive https://github.com/facebookresearch/multimodal.git torchmultimodal
    cd torchmultimodal

    pip install -e .
    ```
    For developers please follow the [development installation](https://github.com/facebookresearch/multimodal/blob/main/CONTRIBUTING.md#development-installation).

## Documentation

The library builds on the following concepts:
- **Architectures**: These are general and composable classes that capture the core logic associated with a family of models. In most cases these take modules as inputs instead of flat arguments (see Models below). Examples include the `LateFusionArchitecture`, `FLAVA` and `CLIPArchitecture`. Users should either reuse an existing architecture or a contribute a new one. We avoid inheritance as much as possible.

- **Models**: These are specific instantiations of a given architecture implemented using builder functions. The builder functions take as input all of the parameters for constructing the modules needed to instantiate the architecture. See [cnn_lstm.py](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/cnn_lstm.py) for an example.

- **Modules**: These are self-contained components that can be stitched up in various ways to build an architecture. See [lstm_encoder.py](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/encoders/lstm_encoder.py) as an example.

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

TorchMultimodal is BSD licensed, as found in the [LICENSE](LICENSE) file.
