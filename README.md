# NSDP

[**Paper**](https://arxiv.org/pdf/2210.05616.pdf) | [**arXiv**](https://arxiv.org/pdf/2210.05616.pdf)  | [**Video**](https://youtu.be/neKuf85H0nE) | [**Project Page**](https://tangjiapeng.github.io/projects/NSDP/) <br>

This is the repository that contains source code for the paper:

**Neual Shape Deformation Priors (NeurIPS 2022 SpotLight).**

- For the task of shape manipulation, NSDP learns shape deformations via canonicalization :
<div style="text-align: center">
<img src="media/pipeline.png" />
</div

- We propose Transformer-based Deformation Networks (TDNets) for local deformation fields prediction :
<div style="text-align: center">
<img src="media/TDNets.png"  />
</div>

## Install all dependencies

- Download the latest conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

- To create a conda environment with all the required packages using conda run the following command:

```
conda env create -f environment.yml
```

> The above command creates a conda environment with the name **nsdp**.


- Compile external dependencies inside `external` directory by executing:

```
conda activate nsdp
./build_external.sh
``` 

> The external dependencies are **PyMarchingCubes**, **gaps** and **Eigen**.

- NSDP uses farthest point sampling (FPS) to downsample the input. Run

```
pip install pointnet2_ops_lib/.
```

in order to install the cuda implementation of FPS. Credits for this go to Erik Wijams's [GitHub](https://github.com/erikwijmans/Pointnet2_PyTorch), from where the code was copied for convenience.


## Data Preparation

In our paper, we mainly use the [DeformingThing4D](https://github.com/rabbityl/DeformingThings4D).
Download the raw dataset firstly, and then convert the .anime files to mesh .obj files by running the script. 

```
cd ./preprocess
bash convert_deform4d_anime_to_mesh.sh
```

We normalize the meshes, sample surface point cloud trajectories (i.e. point clouds with one-to-one correspondences), and sample spatial point trajectories in 3D space (i.e. spreprocess_deform4d_seqpatial points with one-to-one correspondences). To do so, run the script.

```
bash preprocess_deform4d_seq.sh
```

We also use the animations of unseen identities from the [Deformation Transfer](https://people.csail.mit.edu/sumner/research/deftransfer/Sumner2004DTF.pdf) as a test set. To prepare the processed dataset, you can run

```
bash preprocess_deformtransfer_seq.sh
```

To evaluate the user-specified handles used in the interactive editing applications, we use the meshes from [TOSCA](https://vision.in.tum.de/data/datasets/partial) and reconstructed dog using the method of [BARC](https://barc.is.tue.mpg.de/). You can directly download the mesheds of [TOSCA_animal](https://drive.google.com/file/d/1FucECj8VkH-mVcqY7n9HBiKxuXPzrsvT/view?usp=share_link) and [dog_barc_recon](https://drive.google.com/file/d/1_Wote4YCRNH8uj-FPTxaVxNfHQWTVHUh/view?usp=share_link) previously obtained by us, and then run the below scripts to get the normalized meshes. 

```
bash preprocess_nocorr_tosca.sh
bash preprocess_nocorr_dogrec.sh
```

## Pretrained Models
We provide the [pretrained models](https://drive.google.com/file/d/1ghYJKOT7DNiK_20A_goAQZCGj_GnV7q3/view?usp=share_link) of forward and backward deformation networks, and also the pretrained model of whole model after end-to-end finetuning. 

```
unzip pretrained.zip
```

## Training from Scratch

The training is composed of two stages. In the first stage, we train the forward ana backward deformation networks separately using the scripts:

```
python train.py config/deform4d/forward.yaml --with_wandb_logger
python train.py config/deform4d/backward.yaml --with_wandb_logger
```

In the second stage, we train the forward and backward deformation networks together to learn shape deformations between two arbitrary non-rigid poses. 
We need to load the pretrained forward/backward deformation models by modify the ```config['training']['weight_forward_file']``` and ```config['training']['weight_backward_file']``` in ```config/deform4d/arbitrary.yaml```.

```
python train.py config/deform4d/arbitrary.yaml --with_wandb_logger
```
## Evaluation

To evaluate the pretrained model on unseen motions (S1) and unseen identities (S2) of DeformingThing4D, and unseen identies used in the Deformation Transfer you can run

```
python test.py config/deform4d/arbitrary.yaml
python test.py config/deform4d/arbitrary_unseen_iden.yaml
python test.py config/deform4d/arbitrary_unseen_iden.yaml
```

Then, you will get both quantitative and qualitative results.

## User-specified Handle Displacements

To evaluate our approach on user-specified handles of unseen identites, you can run

```
python run.py config/tosca/head.yaml
python run.py config/tosca/tail.yaml
python run.py config/tosca/behindrightfoot.yaml
python run.py config/tosca/frontleftfoot.yaml
python run.py config/dogrec/head.yaml
python run.py config/dogrec/tail.yaml
python run.py config/dogrec/behindrightfoot.yaml
python run.py config/dogrec/frontleftfoot.yaml
```


If you find NSDP useful for your work please cite:

```
@inproceedings{
    tang2022neural,
    title={Neural Shape Deformation Priors},
    author={Tang, Jiapeng and Markhasin Lev and Wang Bi and Thies Justus and Nie{\ss}ner, Matthias},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022},
    }
```

Contact [Jiapeng Tang](mailto:jiapeng.tang@tum.de) for questions, comments and reporting bugs.