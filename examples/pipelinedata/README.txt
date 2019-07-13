SALT2 Training Examples

```
export PYTHONPATH=/Users/David/Dropbox/research/SALT3/salt3/training:$PYTHONPATH
export PATH=/Users/David/Dropbox/research/SALT3/salt3/training:$PATH
TrainSALT.py -c SALT.conf
```

Brief working example for the pipeline:
```
In [2]: from salt3.pipeline.pipeline import *

In [3]: pipe = SALT3pipe(finput='sampleinput_test.txt')

In [4]: pipe.configure()
Adding key COLOR=True in [FLAGS]
Adding key STRETCH=True in [FLAGS]
Adding key DIST_PEAK=0.0 in [COLOR]
input file saved as: pipelinetest/TEST_BYOSED.params
Load base sim input file.. SIMGEN_BYOSEDTEST.INPUT
Adding key NGEN=5
Write sim input to file: pipelinetest/TEST_SIMGEN_BYOSEDTEST.INPUT
Adding key snlist=exampledata/snana/data/SALT3TEST_SIMPLE.LIST in [iodata]
Adding key waverange=2000,9200 in [trainparams]
input file saved as: pipelinetest/TEST_SALT.conf

In [5]: pipe.run()
```
