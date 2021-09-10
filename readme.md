
Code for paper 'Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation'.

# Dataset preparing

Download the [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) (use our provided image list files) dataset. And denote the path of data list in the code. The code is expected to reproduce the results with PyTorch **1.3**. 

# Checkpoint

You can find all the training log files and the weights (before and after the adaptation) on VisDA and Office-Home in this [link](https://drive.google.com/drive/folders/1Tx-iyEXDbmuxlLyYX5sLKwNsTrpwHpjk?usp=sharing). **If you want to reproduce the results quickly, please use the provided source model.**

# VisDA


First train the model on source domain, then do target adaptation without source data:
> python train_src.py
>
> python train_tar.py

# Office-Home
Code for Office-Home is in the 'office-home' folder. 

> sh train_src_oh.sh
>
> sh train_tar_oh.sh

# PointDA-10

Code is based on [PointDAN](https://github.com/canqin001/PointDAN). run the src.sh for source pretraining and tar.sh for source-free domain adaption.
