# Preprocessing for BEST

This ED Producer preprocesses CMS Monte Carlo samples. After preprocessing, these datasets 
can be used to train BEST. In the context of this software package, preprocessing means
reducing the size of the input data set by organizing TTrees by jet, then performing preselection
on those jets and matching to gen particles, and finally calculating and storing only the variables
of interest to BEST.

## Instructions

The actual producer is located in the ``plugins/BESTProducer.cc`` and
the run instructions are located in ``test/run_*.py``. For instructions to run the producer
see the README inside the ``test/`` directory.

