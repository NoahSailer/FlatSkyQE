import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import numpy as np
from EffectiveMask import create_mock
from time import time
import sys

Nsims = 400000

t0 = time()
Nproc = int(sys.argv[1])
p = Pool(Nproc)
p.map(create_mock,np.arange(Nsims))
t1 = time()
print('Took',round(t1-t0,1),'seconds to make',Nsims,'mocks with',Nproc,'processes')