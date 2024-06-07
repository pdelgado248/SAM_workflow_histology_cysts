# SAM-based Automatic Workflow for Histology Cyst Segmentation in Autosomal Dominant Polycystic Kidney Disease

Code from the article "SAM-based Automatic Workflow for Histology Cyst Segmentation
in Autosomal Dominant Polycystic Kidney Disease" (ADPKD) to process a histology image from an ADPKD-affected mouse kidney using a Segment Anything Model (SAM)-based workflow and obtain a segmentation mask of the cysts,
together with an estimate mask of the whole kidney.

This is a schematic representation of the process applied to the image:

![alt text](<SAM application workflow.png>)


## Prerequisites

This code can be run in Python 3.9. It requires the installation of the Segment Anything Model (SAM) available from https://github.com/facebookresearch/segment-anything. It is possible to install it by running 

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

The checkpoint for SAM's pretrained weights has been downloaded from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth. Save this checkpoint ("sam_vit_h_4b8939.pth") in the repository's main folder so that it is referenced by default by our method to apply SAM using its pretrained weights. 

To run SAM efficiently, using a GPU is recommended, hence the CUDA package should be installed if possible to connect to an available GPU. This and other required packages can be installed from the requirements.txt file, corresponding to the packages of the anaconda environment where the code was developed.


## Running the code

### Applying the method

The script "SAM_cyst_workflow.py" contains a class of the same name that includes the whole processing workflow. The jupyter notebook "application_example.ipynb" shows how to call this class in order to segment either a single image or a whole folder of images, to obtain for each of them a cysts mask and a full kidney mask.

It is necessary to provide the path to your image, as well as a path for the pretrained weights to be loaded on SAM, and an output folder to store the results. The example image appearing in the visual schematic is provided within the "Example data" folder. It can be unzipped after cloning the repository to apply the process to it. By default, when running "application_example.ipynb" the input image will be the example image (unzipped in the same folder where the .zip file is), and the output folder will be the empty "Results" folder.

### Scripts to obtain the article's results

Within the folder "Evaluation for the article" are a series of Notebooks that have been used to obtain the results presented in the article. 



