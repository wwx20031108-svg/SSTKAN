# SSTKAN
### 📖[**Paper**] | 

PyTorch codes for "[Spectral-Spatial-Temporal Kolmogorov-Arnold Network for Hyperspectral Change Detection]


### 🌱 Abstract
>Hyperspectral change detection (HCD) aims to recognize altered areas between hyperspectral images captured at different times, which is one of the crucial research areas in remote sensing. In recent years, convolutional neural networks and transformers-based models have been popularly exploited for hyperspectral change detection. However, these models are based on the fixed-weight linear transformation, which struggle to effectively model the intricate spectral-spatial relationships inherent in hyperspectral images. Meanwhile, these methods usually neglect the feature distribution discrepancy induced by different environmental factors. To alleviate these issues, this paper proposes a novel spectral-spatial-temporal Kolmogorov-Arnold Network (SSTKAN) for hyperspectral change detection. First, a spectral-spatial Kolmogorov-Arnold Network is proposed to extract the spectral-spatial features from the input bitemporal hyperspectral images. Then, a 3D Kolmogorov-Arnold Network is exploited to extract the difference and temporal features. Next, a second-order statistical alignment method is proposed to reduce the feature distribution discrepancy of the same ground objects from bitemporal features. Finally, a multi-scale feature enhancement module is designed to strengthen the feature representation by aggregating information from multiple receptive fields followed by a fully connected layer to generate the final detection map. Experiments on four public hyperspectral datasets demonstrate that the proposed SSTKAN outperforms other advanced change detection approaches in both qualitative and quantitative results.
>

### Overall
<div align=center>
<img src="network.png" width="700px">
</div>

### Install
```
git clone https://github.com/XY-boy/FreMamba.git
```

## 🎁 Dataset
Please download the following remote sensing benchmarks:
| Data Type | [AID](https://captain-whu.github.io/AID/) | [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) | [DIOR](https://www.sciencedirect.com/science/article/pii/S0924271619302825) | [NWPU-RESISC45](https://ieeexplore.ieee.org/abstract/document/7891544)
| :----: | :-----: | :----: | :----: | :----: |
|Training | [Download](https://captain-whu.github.io/AID/) | None | None | None |
|Testing | [Download](https://captain-whu.github.io/AID/) | [Download](https://captain-whu.github.io/DOTA/dataset.html) | [Download](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) | [Download](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp)

🚩Please refer to [Dataset Processing](https://github.com/XY-boy/TTST/tree/main/dataload) to build the LR-HR training pairs.
## 📃 Requirements
> * CUDA 11.1
> * Python 3.9.13
> * PyTorch 1.9.1
> * Torchvision 0.10.1
> * causal_conv1d==1.0.0
> * mamba_ssm==1.0.1

## 🧩 Usage
### Test
- **Step I.**  Use the structure below to prepare your dataset, e.g., DOTA, and DIOR.
/xxxx/xxx/ (your data path)
```
/GT/ 
   /000.png  
   /···.png  
   /099.png  
/LR/ 
   /000.png  
   /···.png  
   /099.png  
```
- **Step II.**  Change the `--data_dir` to your data path.
- **Step III.**  Change the `--pretrained_sr` to your pre-trained model path. 
- **Step IV.**  Run the eval_4x.py
```
python eval_4x.py
```

### Train
```
python train_4x.py
```

## Acknowledgement
Our work is built upon [MambaIR](https://github.com/csguoh/MambaIR). Thanks to the author for sharing this awesome work!

