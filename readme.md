# [Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation (NeurIPS 2021)](https://arxiv.org/abs/2110.04202)

Code for our **NeurIPS** 2021 paper 'Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation'. [[project]](https://sites.google.com/view/trustyourgoodfriend-neurips21/) [[paper]](https://arxiv.org/abs/2110.04202) (The codes are based on our [G-SFDA (ICCV 2021)](https://github.com/Albert0147/G-SFDA))

**Note**: In the code, we do not explicitly compute the self-regularization loss (you will find the comment in the code), instead we do not explicitly remove the self features in the nearest neighbor retriving where the occurrence frequency of self feature acts as a dynamic weight.

## Dataset preparing

Download the [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) (use our provided image list files) dataset. And denote the path of data list in the code. The code is expected to reproduce the results with **PyTorch 1.3 with cuda 10.0**. 

## Checkpoint

You can find all the weights (before and after the adaptation, and the results of the logfile may not be the correct one) on VisDA and Office-Home in this [link](https://drive.google.com/drive/folders/1Tx-iyEXDbmuxlLyYX5sLKwNsTrpwHpjk?usp=sharing). **If you want to reproduce the results quickly, please use the provided source model.**

## VisDA


First train the model on source domain, then do target adaptation without source data:
> python train_src.py
>
> python train_tar.py

## Office-Home
Code for Office-Home is in the 'office-home' folder. 

> sh train_src_oh.sh
>
> sh train_tar_oh.sh

## PointDA-10

Code in the folder 'pointDA-10' is based on [PointDAN](https://github.com/canqin001/PointDAN). Run the src.sh for source pretraining and tar.sh for source-free domain adaption.


