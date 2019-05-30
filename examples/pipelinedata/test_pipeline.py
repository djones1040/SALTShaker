##Tests for pipeline
import sys
sys.path.append('/home/mi/salt3/SALT3/salt3')
from pipeline.pipeline import *

def test_pipeline():
    pipe = SALT3pipe(finput='sampleinput_test.txt')
    pipe.configure()
    pipe.run()

test_pipeline()
