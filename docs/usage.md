# Basic Usage                                                                                                                 
## On start                                                                                                                         
After cloning our repo and navigating into the directory open an ipython session and import our package.                 
```python                                                                                                                
ipython      
from DeepSlice import DeepSlice       
```                                                                                                                      
Next, initiate the model and load our weights file.                                                                      
```python                                                                                                                
Model = DeepSlice()         
Model.Build() 
```                                                                             

---
**Important**

* Sections in a folder must all be from the same brain

* DeepSlice uses all the sections you select to inform its prediction of section angle. Thus it is important that you do not include sections which lie outside of the Allen Brain Atlas. This include extremely rostral olfactory bulb and caudal medulla. **If you include these sections in your selected folder it will reduce the quality of all the predictions**.

* The sections do not need to be in any kind of order. 

* The model downsamples images to 299x299, you do not need to worry about this but be aware that there is no benefit from using higher resolutions.

------

## Predictions

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
This will produce an XML file which can be placed in the same directory as your images (here that would be "your_brain_folder") and then opened with QuickNII. 

## Advanced Usage

DeepSlice has several advanced features you may be interested in. DeepSlice first makes an initial pass through which produces independant predictions for each section. In the next stage it goes through and makes all sections parallel. Assuming all sections are from the same brain this second step makes the predictions more accurate. However if you would like to disable this second step simply pass the argument
```python
DeepSlice.predict('parent_dir/', prop_angles=False)
```
if you have alternative weights you would like to use or you have previously trained your own DeepSlice model, these can be passed to the model builder with the command

```python
DeepSlice.Build('path_to_weights/weights.h5')
```

