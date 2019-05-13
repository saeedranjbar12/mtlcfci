# mtlcfci

This repository is for reporoducing the results of the fallowing paper 

```
S. R. Alvar and I. V. Bajić, "Multi-task learning with compressible features for Collaborative Intelligence".
arXiv preprint arXiv:1902.05179.
```
To run the code you need Pytorch = 0.4 and the Python=3.5.2. The corresponding weights are in the [here](https://drive.google.com/drive/folders/1QX26GjOz-j1kjf_tkhth9vdARXhIOSQ0?usp=sharing). The names of the weight files is the encoder type (256 or  512)_ (1=baseline or 2=proposed) 

There are 4 sets of experiments: 
1. Encoder (512x8x16)  baseline 
2. Encoder (512x8x16)  proposed
3. Encoder (256x16x32) baseline 
4. Encoder (256x16x32) proposed

For each experiment you need to run the experiments for 5 times (PNG --quality 0 , JPEG --quality:95,90,85,80). The byte number for the 500 sequences and the average performance will be printed. (./reults includes the printed result and the exel file is the summary of 20 txt files and has the curves plotted).
To see the results of each experiment use a command silimar to the fallowing
```
python ./write_features_3_tasks_512.py --dataset cityscapes --img_rows 256 --img_cols 512 --resume_feature ./feature_model_512_1.pkl --resume_segment ./segment_model_512_1.pkl --resume_reconstruct ./reconstruct_model_512_1.pkl --resume_depth ./depth_model_512_1.pkl --quality 0 
```
For encoder 256x16x32 change 512 to 256.

Cityscapes dataset is needed and you should update the direcotry of the dataset in the config.jason file. 

After the results obtained, you can use the bjontegaard2.m to get the results in table 1. 

References:

Parts of the code is based on the codes from:

https://github.com/meetshah1995/pytorch-semseg

https://www.mathworks.com/matlabcentral/fileexchange/41749-bjontegaard-metric-calculation-bd-psnr