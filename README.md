# DeepSlice
DeepSlice is a python library which automatically aligns mouse histology with the allen brain atlas common coordinate framework.
The files are viewable using the [QuickNII](https://www.nitrc.org/projects/quicknii "QuickNII") software package.
DeepSlice requires no preprocessing and works on any stain, however we have found it performs best on brightfield images.

## Install

```
conda create env -f DeepSlice.yaml
```
## Usage

After cloning our repo and navigating into the directory open an ipython session and import our package.
```python
ipython
from DeepSlice import DeepSlice
```
Next, initiate the model and load our weights file.
```python
Model = DeepSlice("Synthetic_data_final.hdf5")
Model.Build("xception_weights_tf_dim_ordering_tf_kernels.h5")
```
Now your model is ready to use, just direct it towards the parent directory of the file you're interested in.
<br/> eg:
```bash
├── parent_dir
│   ├── your_brain_folder
│   │   ├── brain_slice_1.png
│   │   ├── brain_slice_2.png
│   │   ├── brain_slice_3.png
```
In this parent directory there should be only one sub folder, in this example this is "your_brain_folder".
<br />To align these images using DeepSlice simply call
```python
Model.predict('parent_dir/')
Model.Save_Results('output_filename')
```
This will produce an XML file which can be placed in "parent_dir" and then opened with QuickNII. 



















