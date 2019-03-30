# Training HHESTIA

The programs in this directory train the HHESTIA.

## Instructions for HHESTIA without images

In order to run the training program, simply use the CMS environment to run ``trainHHESTIA.py``.

```bash
cd HHESTIA/training/
cmsenv
python trainHHESTIA.py
```

## Overview

``trainHHESTIA.py`` has can produce plots of the input variables and training results. These features can
be turned on and off with the boolean variables at the beginning of the program. 

The tools directory contains 
some useful modules for use with training the neural network. Of particular importance is ``getBranchNames()``
located in ``functions.py``. This function uses several if statements in order to ignore branches so that they
don't get included in the training. To ignore more variables, simple add to these if statements.

# Image Instructions

Eventually, the images will be added to HHESTIA, but for now they are seperate. Images must first be created and
then can be trained over.

## Image Creation

The images can be created with the appropriate .root files from the preprocessing step. To create the images, run
one of the ``imageCreator.py`` files in the CMS environment.

```bash
cd HHESTIA/training/
cmsenv
python imageCreator.py
```

## Training Convnet 

The images can be used to train a convolutional neural network. To do so, set up an appropriate GPU environment. 
Do NOT use the CMS environment. This first set of instructions is for setting up the GPU environment at Fermilab's 
LPC.

```bash
/bin/bash --login
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source activate mlenv3
```

Now, a network can be trained on the images. Various network options are available in the files titled ``imageTraining.py``.

```bash
cd HHESTIA/training/
python imageTraining.py
```


## Warning About Functions in Python

Python does not forget about operations done to a variable inside a function. If a variable ``var`` is declared
in the main program and a function then deletes ``var`` in order to return something else, ``var`` will also be
deleted from the main program. This also includes any variables that point to the same memory; for example 
``var2 = var`` will also be deleted. To avoid this, use the copy module to copy the memory.

```python
import copy
var2 = copy.copy(var)
result = function(var) # function that deletes var
# var2 will still be here, but var will be deleted
```

