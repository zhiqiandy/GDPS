## Getting started

#### 1. Prepare the Environment

- python 3.8
- PyTorch 2.3
- CUDA 12.1

Lower version of PyTorch with proper CUDA should work but not be fully tested.

```
# in GDPS folder

conda create -n GDPS python=3.8
conda activate GDPS

pip install -r requirements.txt

# (optional) install PyTorch with proper CUDA
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

We use [bkse](https://github.com/VinAIResearch/blur-kernel-space-exploring) for nonlinear blurring and [motionblur](https://github.com/LeviBorodenko/motionblur) for motion blur. **No further action required then**.



#### 2. Prepare the pretrained checkpoint

Download the public available FFHQ checkpoint (ffhq_10m.pt) [here](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh).

```
# in GDPS folder

mkdir checkpoint
mv {DOWNLOAD_DIR}/ffqh_10m.pt checkpoint/ffhq256.pt
```



**(Optional)** For nonlinear deblur task, we need the pretrained model from [bkse](https://github.com/VinAIResearch/blur-kernel-space-exploring) at [here](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view?usp=drive_link):

```
# in GDPS folder

mv {DOWNLOAD_DIR}/GOPRO_wVAE.pth forward_operator/bkse/experiments/pretrained
```



#### 3.  (Optional) Prepare the dataset (or use provided examples)

You can add any FFHQ256 images you like to `dataset/demo` folder



#### 4. Sample

Make a folder to save results:

```
mkdir results
```

##### Phase Retrieval

Now you are ready for run. For **phase retrieval** with GDPS-1k in 4 runs for $10$ demo images in `dataset/demo`:

```
python posterior_sample.py \
+data=demo \
+model=ffhq256ddpm \
+task=phase_retrieval \
+sampler=edm_daps \
save_dir=results \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=10 \
data.start_id=0 data.end_id=10 \
name=phase_retrieval_demo \
gpu=0
```

The results are saved at foloder `\results`. 



##### All Tasks

```
python posterior_sample.py \
+data=demo \
+model=ffhq256ddpm \
+task={TASK_NAME} \
+sampler=edm_daps \
save_dir=results \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=5 \
data.start_id=0 data.end_id=5 \
name={SUB_FOLDER_NAME} \
gpu=0
```

replace the {TASK_NAME} by one of following:

* `phase_retrieval`: phase retrival of oversample ratio of $2.0$

* `down_sampling`: super resolution ($\times 4$)

* `inpainting`:  128x128 box inpainting

* `inpainting_rand`: $70\%$ random inpainting 

* `gaussian_blur`: gaussian deblur of kernel size $61$ and intensity $3$

* `motion_blur`: gaussian deblur of kernel size $61$ and intensity $0.5$

* `nonlinear_blur`: nonlinear deblur of default setting in bkse repo

* `hdr`: high dynamic range reconstruction of factor $2$ 

