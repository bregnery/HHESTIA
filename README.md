# HHESTIA

## Dependencies

This program requires the use of the [Jet Toolbox](https://github.com/cms-jet/JetToolbox/tree/master).
I have included instructions for downloading this in the Installation section.

## Installation

This program is written for use with ``CMSSW_9_X``. Start installation by installing CMSSW.

```bash
cmsrel CMSSW_9_4_8
cd CMSSW_9_4_8/src/
scram b -j8
```
Now, add the Jet Toolbox.

```bash
git clone -b jetToolbox_94X https://github.com/cms-jet/JetToolbox JMEAnalysis/JetToolbox
scram b
cmsRun JMEAnalysis/JetToolbox/test/jettoolbox_cfg.py
```

Then, clone this repository and compile the programs as modules for CMSSW.

```bash
cd CMSSW_9_4_8/src/
git clone https://github.com/bregnery/HHESTIA.git
scram b -j8
```

Now the program can be used. 

## Overview

Before training the neural network, the CMS datasets must be converted into a usable form.
To do this, see the instructions in the ``preprocess`` directory.
After preprocessing, the files can be used to train a neural network. For this process,
see the instructions in the ``training`` directory.

