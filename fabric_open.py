#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import math as math
from math import sin
from decimal import Decimal

#print np.version.version

# Open Strain Data
a=open(sys.argv[1])
strain_data=[]
for line in a:
	strain_line = a.readline().split()
	strain_data.append(strain_line)


# Open Fabric Data
fabric_data=[]
b=open(sys.argv[2])
b.readline()
b.readline()
b.readline()
b.readline()
for line in b:
	fabric_line = b.readline().split()
	fabric_data.append(fabric_line)



