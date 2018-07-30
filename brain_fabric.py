#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import math as math
from math import sin
from decimal import Decimal

#print np.version.version

# Open Strain Data

strain_data = np.loadtxt(sys.argv[1], delimiter=' ')

# Open Fabric Data

fabric_data = np.genfromtxt(sys.argv[2], delimiter=(8, 8,8,6,6,6,6,6,3,5,3, 3), skip_header=(8))
