# Installation

**First** To use DeepLabCut you must have python3 installed. In order to easily install all the dependancies we recommend using [Anaconda](https://www.anaconda.com/products/individual "Anaconda Installation Files"). 


**Second** Once anaconda is installed, cd into your cloned DeepSlice directory, then cd into the 'conda_environments' directory, and use our premade environment files to setup your system. 
```
cd DeepSlice/conda_environments
```
* **CPU Installation** For most users we recommend using the DS-CPU.yaml installation file. this will install all the dependencies required to run DeepSlice on your CPU. 
Do this with the command: 

      conda env create -f DS-CPU.yml


* **GPU Installation** If you wish to run DeepSlice on a huge number of images, and have access to an nvidia GPU then please use the DS-GPU.yaml installation file.

      conda env create -f DS-GPU.yml

**Finished :)** You are now ready to run DeepSlice. Just activate the environment using 
```python
conda activate DS-CPU
```
or 
```python 
conda activate DS-GPU
```
If you run into any problems create a github issue and I will help you solve it.
