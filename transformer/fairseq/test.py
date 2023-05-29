import sys
sys.path.append('D:\\13Research\\AI\\SolarEnergyMaterials\\')
import  argparse
import fairseq
from task import register_task
from fairseq.tasks import FairseqTask
from transformer.dataset import ogbGraphormerDataset

import importlib
importlib.reload(ogbGraphormerDataset)

# dataset_spec either "ogbg-molhiv", "ogbg-molpcba", "pcqm4mv2", "pcqm4m"
test = ogbGraphormerDataset.ogbGraphDataset(dataset_name= "pc2qm4mv2", seed=0)
print(type(test))