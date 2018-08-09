# Preprocessing for training HHESTIA

This ED Producer preprocesses CMS Monte Carlo samples. After preprocessing, these datasets 
can be used to train HHESTIA.

## Instructions

The actual producer is located in the ``plugins/HHESTIAProducer.cc`` and
the run instructions are located in ``test/run_*.py``. To run, use the
cms environment to run a ``run_*.py`` file. For example: 

```bash
cd BESTHHedmanalyzer/BESTHHedmanalyzer/test/
cmsenv
cmsRun run_HH.py
```

Be sure to update any file locations in the ``run_*.py`` files!!


