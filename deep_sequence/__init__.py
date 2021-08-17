# Ammend path to include the DeepSequence submodule. Better would be to have
# this installed to PyPI, but it works
import os 
import sys

# Get the directory of the deep sequence module and add it to path
DEEPSEQ_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "DeepSequence")
sys.path.append(DEEPSEQ_DIR_PATH)