# ClusterNet

This is a official implementation of "Online Unsupervised Video Object Segmentation via Contrastive Motion Clustering" (IEEE TCSVT).

<div align="center">
  <img width="600", src="./figs/framework.png">
</div>

Papers: [[TCSVT](https://ieeexplore.ieee.org/document/10159996)] [[arXiv](https://arxiv.org/abs/2306.12048)]

## Prerequisites

The training and testing experiments are conducted using PyTorch 1.8.1 with a single NVIDIA TITAN RTX GPU with 24GB Memory.

- python 3.8
- pytorch 1.8.1
- torchvision 0.9.1

Other minor Python modules can be installed by running
```bash
pip install opencv-python einops
```

## Datasets

- [*DAVIS~16~*](https://davischallenge.org/davis2017/code.html#unsupervised): We perform online clustering and evaluation on the validation set. However, please download *DAVIS~17~* (Unsupervised 480p) to fit the code.
- [*FBMS*](https://lmb.informatik.uni-freiburg.de/resources/datasets/): This dataset contains videos of multiple moving objects, providing test cases for multiple object segmentation.
- [*SegTrackV2*](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html): Each sequence contains 1-6 moving objects. 

Following the evaluation protocol in [CIS](https://arxiv.org/abs/1901.03360), we combine multiple objects as a single foreground and use the region similarity $\mathcal{J}$ to measure the segmentation performance for the *FBMS* and *SegTrackV2*. Binary Mask: [[FBMS](https://drive.google.com/file/d/16zzb10mVNuRAC3lrJ984jxWthcTWqXvl/view?usp=sharing)][[SegTrackV2](https://drive.google.com/file/d/1twATOkSw7D3ZyH7wLmwApF8WL_-jhh9m/view?usp=sharing)]
- Path configuration: Dataset path settings is ```--data_dir``` in ```train_sequence_BGC.py```.
```python
parser.add_argument('--data_dir', default=None, type=str, help='dataset root dir')
```

- The datasets directory structure will be as follows:
```text
|--DAVIS2017
|   |--Annotations_unsupervised
|   |   |--480p
|   |--ImageSets
|   |   |--2016
|   |--Flows_gap_1_${flow_method}
|       |--Full-Resolution
|--FBMS
|   |--Annotations_Binary
|   |--Flows_gap_1_${flow_method}
|--SegTrackv2
    |--Annotations_Binary
    |--Flows_gap_1_${flow_method}
```

## Precompute optical flow

- The optical flow is estimated by using the [PWCNet](https://github.com/NVlabs/PWC-Net), [RAFT](https://github.com/princeton-vl/RAFT) and [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official). In datasets directory, the variable ```flow_method``` is ```PWC```, ```RAFT``` and ```FlowFormer```, respectively.

- The flows are resized to the size of the original image (same as [Motion Grouping](https://github.com/charigyang/motiongrouping)), with each input frame having a size of $480\times854$ for the *DAVIS~16~* and $480\times640$ for the *FBMS* and *SegTrackV2*. We convert the optical flow to 3-channel images with the standard visualization used for the optical flow and normalize it to $[-1, 1]$, and use only the previous frames for the optical flow estimation in the online setting.

## Train & Inference

To train the ClusterNet model on a GPUs, you can use:
```bash
bash scripts/train_sequence_BGC.sh
```

In the ```train_sequence_BGC.sh``` file, first activate your Python environment and set ```gpu_id``` and ```data_dir```. Then set the hyperparameters ```batch_size```, ```n_clusters```, and ```threshold``` to 16, 30, and 0.1, respectively.

## Outputs

The model files and checkpoints will be saved in ```./checkpoints/${exp_id}```.

```.pth``` files with ```_${sequence_name}``` store the network weights that initialize our autoencoder to train on *DAVIS~16~* through the loss of optical flow reconstruction.

The segmentation results will be saved in ```./results/${exp_id}```. The evaluation criterion is the mean region similarity $\mathcal{J}$.

| Optical flow prediction | Method | Mean $\mathcal{J}\uparrow$ |
|:--:|:--:|:--:|
|[PWC-Net](https://arxiv.org/abs/1709.02371)|[MG](https://arxiv.org/abs/2104.07658)<br>ClusterNet|63.7<br>67.9(+4.2)|
|[RAFT](https://arxiv.org/abs/2003.12039)|[MG](https://arxiv.org/abs/2104.07658)<br>ClusterNet|68.3<br>72.0(+3.7)|
|[FlowFormer](https://arxiv.org/abs/2203.16194)|[MG](https://arxiv.org/abs/2104.07658)<br>ClusterNet|70.3<br>75.4(+5.1)|

## Citation

If you find our work useful in your research please consider citing our paper!

```bib
@ARTICLE{ClusterNet,
  author={Xi, Lin and Chen, Weihai and Wu, Xingming and Liu, Zhong and Li, Zhengguo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Online Unsupervised Video Object Segmentation via Contrastive Motion Clustering}, 
  year={2023}
```

## Contact
If you have any questions, please feel free to contact Lin Xi (xilin1991@buaa.edu.cn).

## Acknowledgement
This project would not have been possible without relying on some awesome repos: [Motion Grouping](https://github.com/charigyang/motiongrouping), [PWCNet](https://github.com/NVlabs/PWC-Net), [RAFT](https://github.com/princeton-vl/RAFT) and [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official). We thank the original authors for their excellent work.