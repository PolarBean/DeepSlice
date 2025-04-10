

[![DOI](https://zenodo.org/badge/274122364.svg)](https://zenodo.org/badge/latestdoi/274122364)
![PyPI - Version](https://img.shields.io/pypi/v/DeepSlice)
![PyPI - Downloads](https://img.shields.io/pypi/dm/DeepSlice)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/DeepSlice)


![Alt](docs/images/DeepSlice_github_banner.png "DeepSlice Banner")
DeepSlice is a python library which automatically aligns mouse histology with the allen brain atlas common coordinate framework (and now rat brain histology to the Waxholm rat brain atlas, though this is in beta).
The alignments are viewable, and refinable, using the [QuickNII](https://www.nitrc.org/projects/quicknii "QuickNII") software package.
DeepSlice requires no preprocessing and works on any stain, however we have found it performs best on brightfield images.
At present one limitation is that it only works on Coronally cut sections, we will release an update in the future for sagittal and horizontally cut histology.
![Alt](docs/images/process.PNG) 
DeepSlice automates the process of identifying exactly where in the brain a section lies, it can accomodate non-orthogonal cutting planes and will produce an image specific annotation for each section in your brain.  
## Citation
If you use DeepSlice in your work please cite [Carey et al, 2023](https://www.nature.com/articles/s41467-023-41645-4). It may also be useful if you mention the version you use :)

In addition, you should also remember to cite [Wang et al, 2020](https://doi.org/10.1016/j.cell.2020.04.007) if you use the Allen CCFv3 atlas for the Mouse model and [Kleven et al, 2023](https://www.nature.com/articles/s41592-023-02034-3) if you use the Waxholm Atlas of the Sprague Dawley Rat for the Rat model.

## Workflow 
DeepSlice is fully integrated with the <a href="https://quint-workflow.readthedocs.io/en/latest/QUINTintro.html" >QUINT workflow.</a>  Quint helps you register, segment and quantify brain wide datasets! &nbsp; 🐭🧠🔬💻🤖

## Web Application
If you would like to use DeepSlice but don't need your own personal installation, check out [**DeepSlice Flask**](https://www.DeepSlice.com.au), a web application which will allow you to upload your dataset and download the aligned results. Some more advanced options are only available in the Python package. The web interface was developed by [Michael Pegios](https://github.com/ThermoDev/).
## [Installation: How to install DeepSlice](#installation)

## [Usage: How to align using DeepSlice](#basic-usage)
## [For a jupyter notebook example check out](examples/example_notebooks/DeepSlice_example.ipynb)

**Happy Aligning :)**


<br>


<a name='Installation'></a> 
<h1> Installation </h1>
<!-- This h2 must be bold  -->

<h2 style="font-weight: bold; text-decoration: underline"> From PIP  </h2>
This is the easy and recommended way to install DeepSlice, first make sure you have Python 3.11 installed and then simply:

```bash
pip install DeepSlice
```
And you're ready to go! 🚀 Check out the PyPi package [here](https://pypi.org/project/DeepSlice/)

If you run into any problems create a github issue and I will help you solve it.

<br>

<a name='BasicUsage'></a>    
# Basic Usage                                                                                                         
## On start                                                                                                                         
After cloning our repo and navigating into the directory open an ipython session and import our package.                 
```python                                                                                                                
from DeepSlice import DSModel     
```                                                                                                                      
Next, specify the species you would like to use and initiate the model.                                                                    
```python                                                                                                                
species = 'mouse' #available species are 'mouse' and 'rat'

Model = DSModel(species)
```                                                                             

---
**Important**

* Sections in a folder must all be from the same brain

* DeepSlice uses all the sections you select to inform its prediction of section angle. Thus it is important that you do not include sections which lie outside of the Allen Brain Atlas. This include extremely rostral olfactory bulb and caudal medulla. **If you include these sections in your selected folder it will reduce the quality of all the predictions**.
* If you are not using the web version and would like to include these sections in your alignment, you can now label them as "bad sections" (see below), which will tell DeepSlice not to weight these sections in the propagation.

* The sections do not need to be in any kind of order. 

* The model downsamples images to 299x299, you do not need to worry about this but be aware that there is no benefit from using higher resolutions.

------

## Predictions

Now your model is ready to use, just direct it towards the folder containing the images you would like to align.            
<br/> eg:                                                                                                                
```bash                                                                                                              
    
 ├── your_brain_folder
 │   ├── brain_slice_1.png 
 │   ├── brain_slice_2.png     
 │   ├── brain_slice_3.png
```                                                                                                                      
In this parent directory there should be only one sub folder, in this example this is "your_brain_folder".               
<br />To align these images using DeepSlice simply call                                                                  
```python                                                                                                                
folderpath = 'examples/example_brain/GLTa/'
#here you run the model on your folder
#try with and without ensemble to find the model which best works for you
#if you have section numbers included in the filename as _sXXX specify this :)
Model.predict(folderpath, ensemble=True, section_numbers=True)    
#This is an optional stage if you have damaged sections, or hemibrains they may negatively effect the 
#propagation for the entire dataset simply set the bad sections here using a string which is unique to 
#those each section you would like to label as bad. DeepSlice will not include it in the propagation 
#and instead it will infer its position based on neighbouring sections.
Model.set_bad_sections(bad_sections=["_s094", "s199"])
#If you would like to normalise the angles (you should)
Model.propagate_angles()                     
#To reorder your sections according to the section numbers 
Model.enforce_index_order()    
#alternatively if you know the precise spacing (ie; 1, 2, 4, indicates that section 3 has been left out.    
#Furthermore if you know the exact section thickness in microns this can be included instead of None
#if your sections are numbered rostral to caudal you will need to specify a negative section_thickness      
Model.enforce_index_spacing(section_thickness = None)
#now we save which will produce a json file which can be placed in the same directory as your images 
#and then opened with QuickNII. 
Model.save_predictions(folderpath + 'MyResults')                                                                                                             



```
## Acknowledgements
We are grateful to Ann Goodchild for her time-saving blunt assessments of many failed prototypes, for the motivation provided by Dr William Redmond, and especially to Veronica Downs, Freja Warner Van Dijk and Jayme McCutcheon, whose Novice alignments were instrumental to this work. We would like to thank Gergely Csúcs for providing his expertise and many atlasing tools. Work in the authors’ laboratories is supported by the National Health & Medical Research Council of Australia, the Hillcrest Foundation, and Macquarie University (SMcM), and from the European Union’s Horizon 2020 Framework Program for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3) and the Research Council of Norway under Grant Agreement No. 269774 (INCF, JGB). We are grateful to Macquarie University for access to their HPC resources, essential for production of early DeepSlice prototypes.



