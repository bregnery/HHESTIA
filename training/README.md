# Training HHESTIA

The programs in this directory train the HHESTIA.

## Instructions

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

