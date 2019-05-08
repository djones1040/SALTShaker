##Tests for pipeline
import sys
sys.path.append('/home/mi/salt3/SALT3')
from salt3.pipeline.pipeline import *

def test_pipeline():
    pipe = SALT3pipe(finput='examples/pipelinedata/sampleinput.txt')
    pipe.configure()
    pipe.run()