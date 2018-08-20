# Instructions for Preprocessing

The preprocessing program can be run locally or through CRAB.

## Local Instructions

To run, use the cms environment to run a ``run_*.py`` file. For example: 

```bash
cd HHESTIA/preprocess/test/
cmsenv
cmsRun run_HH.py
```

Be sure to update any file locations in the ``run_*.py`` files!!

## CRAB Instructions

First, set up the CRAB environment and obtain a proxy

```bash
cd HHESTIA/preprocess/test/
cmsenv
CRAB
vprox
``` 

The CRAB and vprox commands are aliases see [my bash profile](https://github.com/bregnery/Settings/blob/master/lxplus/.bash_profile).

Now submit any of the CRAB files.

```bash
crab submit submit_crab_*.py
```

The output file should be ``preprocess_HHESTIA_*.root``. DAS datasets can be updated inside the ``submit_crab_*.py`` files.

### Useful CRAB Commands

To test, get estimates, and then submit do a crab dry run

```bash
crab submit --dryrun submit.py
crab proceed
```

To resubmit failed jobs

```bash
crab resubmit crab_projecs/<project_directory>
```

To view job status go to: https://dashb-cms-job-task.cern.ch
 
