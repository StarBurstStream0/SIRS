## The offical PyTorch code for paper 
[""SIRS: Multi-task Joint Learning for Remote Sensing Foreground-entity Image-text Retrieval"", TGRS 2024.]()

##### Author: Zicong Zhu

![Supported Python versions](https://img.shields.io/badge/python-3.7-blue.svg)
![Supported OS](https://img.shields.io/badge/Supported%20OS-Linux-yellow.svg)
![npm License](https://img.shields.io/npm/l/mithril.svg)
<a href="https://pypi.org/project/mitype/"><img src="https://img.shields.io/pypi/v/mitype.svg"></a>

```bash
#### News:
#### 2024.05.11: SIRS and RSITMD-SS is expected to be released after the paper open to access.
```

## INTRODUCTION

We propose a novel Semantic-guided Image-text Retrieval framework with Segmentation (SIRS).

It is a multi-task joint learning framework for plug-and-play and end-to-end training RS CIR models efficiently, including Semantic-guided Spatial Attention (SSA) and Adaptive Multi-scale Weighting (AMW) modules.

##
## SIRS
### Network Architecture

The SIRS architecture takes images, masks, and text as input. 
Segmentation and similarity results are derived after passing through SSA and AMW, which are both plug-and-play.

##
SSA consists of two parts, namely the BR branch and the SS branch.
The former is to perceive background noise, while the latter learns foreground categories.

##
The module structure of AMW.
It is based on the ideas of feature pyramid and intermediate supervision and performs many-to-many similarity calculations with text features.

##
## RSITMD-SS
### Dataset Features

### Segment visualization


<!-- ## Citation
If you feel this code helpful or use this code or dataset, please cite it as
```

``` -->
