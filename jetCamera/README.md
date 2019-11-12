# Jet Camera

## Overview

The Jet Camera is used to produce boosted jet images and store BES variables that are later used to train 
the BEST neural network

The `tools` directory contains some useful modules. Of particular importance is ``getBranchNames()``
located in ``BESfunctions.py``. This function uses several if statements in order to ignore branches so that they
don't get included in the training. To ignore more variables, simple add to these if statements.

# Camera Instructions

Boosted jet images must first be created and then can be trained over.

## Image Creation

The images can be created with the appropriate .root files from the preprocessing step. To create the images, run
one of the ``imageCreator.py`` files in the CMS environment.

```bash
cmsenv
python imageCreator.py
```

