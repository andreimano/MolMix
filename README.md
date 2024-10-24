# MolMix

**Official repository for** [**"MolMix: A Simple Yet Effective Baseline for Multimodal Molecular Representation Learning"**](https://arxiv.org/abs/2410.07981), accepted at the Machine Learning for Structural Biology Workshop, NeurIPS 2024.

## Installation

To install the required dependencies, use the following commands:

```bash
conda create -n molmix python=3.10
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install ogb rdkit schedulefree wandb multimethod matplotlib
pip install flash-attn --no-build-isolation
```

## MARCEL Experiments

To run the MARCEL experiments:

1. Download the `Drugs.zip` and `Kraken.zip` archives from the [official MARCEL repo](https://github.com/SXKDZ/MARCEL).
2. Place them in the following locations:
   - `./datasets/drugs/raw/Drugs.zip`
   - `./datasets/kraken/raw/Kraken.zip`

## Running Experiments

Experiment configuration files are available in the `./cfgs` folder. To run an experiment using a specific configuration file, execute the following:

```bash
python main.py --cfg ./cfgs/THE_SPECIFIC_CONFIG.yaml
```

## Codebase

This repository includes modified implementations from:
- [GemNet](https://github.com/TUM-DAML/gemnet_pytorch)
- [SchNet](https://github.com/atomistic-machine-learning/SchNet)

Additionally, various parts of the code are adapted from the [MARCEL repository](https://github.com/SXKDZ/MARCEL).

## Citation

If you use this work, please cite the following:

```bibtex
@misc{manolache2024molmix,
      title={MolMix: A Simple Yet Effective Baseline for Multimodal Molecular Representation Learning}, 
      author={Andrei Manolache and Dragos Tantaru and Mathias Niepert},
      year={2024},
      booktitle={Machine Learning for Structural Biology Workshop, NeurIPS 2024},
      url={https://arxiv.org/abs/2410.07981}, 
}
```

## Contact

If you have any issues with running the code or the implementation, please either open a GitHub issue, or contact me at 

```bash
echo "manogm.irli.tto-nurei.stturtttdg.de" | sed 's/-/@/1; s/nurei.stturtttdg/ki.uni-stuttgart/; s/manogm.irli.tto/andrei.manolache/'
```
