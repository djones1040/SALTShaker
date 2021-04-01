##Tests for pipeline
import sys
sys.path.append('/home/mi/salt3/SALT3')
from saltshaker.pipeline.pipeline import *

def test_pipeline():
    pipe = SALT3pipe(finput='testdata/pipeline/pipeline_test.txt')
    pipe.build(data=False,mode='customize',onlyrun=['sim','lcfit'])
    pipe.configure()
    pipe.glue(['sim','lcfit'],on='phot')
    pipe.run()
