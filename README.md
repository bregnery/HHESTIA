# BEST: Boosted Event Shape Tagger

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
git clone -b jetToolbox_91X https://github.com/cms-jet/JetToolbox JMEAnalysis/JetToolbox
scram b
cmsRun JMEAnalysis/JetToolbox/test/jettoolbox_cfg.py
```

Then, clone this repository and compile the programs as modules for CMSSW.

```bash
cd CMSSW_9_4_8/src/
git clone https://github.com/boostedeventshapetagger/BEST.git
scram b -j8
```

Now the program can be used. 

## Overview

Before training the neural network, the CMS datasets must be converted into a usable form.
To do this, see the instructions in the ``preprocess`` directory.
After preprocessing, the files can be used to train a neural network. For this process,
see the instructions in the ``training`` directory.

## Instructions for Contributing to this Repository

First, fork this repository and push code to the forked version.
Please only submit pull requests to the `developer` branch. Before submitting a pull request, 
please test your code. To test any changes to the edproducer, please do the following:

```bash
cd BEST/preprocess/
cmsenv
scram b -j8
cmsRun test/run_TEST.py
```

Then open up the output root file and make sure that the results are as expected. There are no
tests yet for any of the training files. So please keep old, stable training code in the `legacy` 
folder and create a new file when you make changes. 

After tests, please rebase to the current developer version:

```bash
# if this is your first time submitting a pull request, then do
git remote add BEST git@github.com:boostedeventshapetagger/BEST.git
git checkout -t BEST/developer
# then every time you want to ensure that the code is up to date
git fetch -p --all
git checkout developer
git pull
git checkout <my feature branch>
git rebase -i developer
```

Finally, submit your a pull request on GitHub to the `developer` branch in `boostedeventshapetagger/BEST`.
There is a short form to fill out for the pull request, this will help the maintainers understand your changes.
Then, your changes will be reviewed before being added. 

