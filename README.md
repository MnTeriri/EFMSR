# Enhanced Fourier-Mixture Transformer for High-Performance Image Super-Resolution (EFMSR), The Visual Computer

## Dependencies

- Ubuntu 24.04
- Python 3.12
- PyTorch 2.8.0 + cu129
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
git clone https://github.com/MnTeriri/EFMSR.git
conda create -n EFMSR python=3.12
conda activate EFMSR
pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Training

- Download [training](https://www.kaggle.com/datasets/anvu1204/df2kdata/data) (DF2K) and [testing](https://github.com/MnTeriri/EFMSR/releases/tag/v1.0) (Set5, Set14, BSD100, Urban100, Manga109) datasets, place them in `datasets/`.

- Run the following scripts. The training configuration is in `options/train/`.

  ```bash
  # EFMSR 
  torchrun --nproc-per-node=8 --master-port=4321 basicsr/train.py -opt options/train/train_EFMSR_x2.yml --launcher pytorch
  torchrun --nproc-per-node=8 --master-port=4321 basicsr/train.py -opt options/train/train_EFMSR_x3.yml --launcher pytorch
  torchrun --nproc-per-node=8 --master-port=4321 basicsr/train.py -opt options/train/train_EFMSR_x4.yml --launcher pytorch
  
  # EFMSR-light
  torchrun --nproc-per-node=8 --master-port=4321 basicsr/train.py -opt options/train/train_EFMSR_Light_x2.yml --launcher pytorch
  torchrun --nproc-per-node=8 --master-port=4321 basicsr/train.py -opt options/train/train_EFMSR_Light_x3.yml --launcher pytorch
  torchrun --nproc-per-node=8 --master-port=4321 basicsr/train.py -opt options/train/train_EFMSR_Light_x4.yml --launcher pytorch
  ```
- The training experiment is in `experiments/`.

## Testing

- We provide EFMSR and EFMSR-light with scale factors: x2, x3, x4.

- Download testing (Set5, Set14, BSD100, Urban100, Manga109) datasets, place them in `datasets/`.

- Run the following scripts.
  ```bash
  # EFMSR
  python basicsr/test.py -opt options/test/test_EFMSR_x2.yml
  python basicsr/test.py -opt options/test/test_EFMSR_x3.yml
  python basicsr/test.py -opt options/test/test_EFMSR_x4.yml
  
  # EFMSR-light
  python basicsr/test.py -opt options/test/test_EFMSR_Light_x2.yml
  python basicsr/test.py -opt options/test/test_EFMSR_Light_x3.yml
  python basicsr/test.py -opt options/test/test_EFMSR_Light_x4.yml
  ```

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR). The smm-cuda is derived from [PFT-SR](https://github.com/LabShuHangGU/PFT-SR/tree/master/ops_smm).