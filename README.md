# SecondPose

Code for "SecondPose: SE(3)-Consistent Dual-Stream Feature Fusion for Category-Level Pose Estimation", Preprint. [[Arxiv](https://arxiv.org/abs/2311.11125)]




## Requirements
The code has been tested with
- python 3.9
- pytorch 1.13.0
- CUDA 11.6

Other dependencies:

```
requirements.txt
```

Setup env:

```
conda create -n secondpose python=3.9
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
cd lib/pointnet2/
pip install .
cd ../sphericalmap_utils/
pip install .
cd ../../
pip install -r requirements.txt
pip install open3d
```

## Data Processing

1. Please refer to the work of [Self-DPDN](https://github.com/JiehongLin/Self-DPDN).
2. run data_preprocess.py


## Network Training


Train SecondPose for rotation estimation:

```
python train_geodino.py --gpus 0 --mod r
```

Train the network of [pointnet++](https://github.com/charlesq34/pointnet2) for translation and size estimation:

```
python train.py --gpus 0  --mod ts 
```


## Evaluation

To test the model, please run:

```
python test_geodino.py --gpus 0 --test_epoch [YOUR EPOCH]
```

## Acknowledgements

Our implementation leverages the code from [VI-Net](https://github.com/JiehongLin/VI-Net), 

## License
Our code is released under MIT License (see LICENSE file for details).

## Contact
`yamei.chen@tum.de`

