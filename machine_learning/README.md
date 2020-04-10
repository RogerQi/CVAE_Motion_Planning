# Roger's Deep Learning Tools of the Trade

This codebase should be as generic as possible. Implementation of fancy ideas should be built
in fork of this repo rather than working in the repo itself to allow rapid development.

Dependencies
- PyTorch (torch, torchvision)
- yacs
- OpenCV
- PIL

TODOs
- Support for 2D segmentation task
- Add better print statement for train summary between epochs
- Add more options for classifier/dataset/loss/network
    - Classifier
        - Dense
    - Dataset
        - Cifar10
        - Coco
    - Loss
    - Network
        - Resnet
- Update README docs in each folder
- Add model saving logic
    - Computation Graph + Weights
    - Just weights
    - Feature Extraction
- Add transforms foolproof sanity checker
    - consistency of normalization in train/test set
    - normalization should only happen at the end of transforms
    - crop_size/input_size consistency check

Future TODOs
- Support for better Logging/Timing (tensorboard?)
- Add closed-loop experiment logic
    - Use deterministic CUDA Ops from [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
    - Fix seeds for all random Ops from [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
    - Save entire EXP folders/config files to backup location
- Add pretrained weights loading logic and backbone freezing logic (support for fine-tuning)
- Support for Detection
- Support for 3D CV task
