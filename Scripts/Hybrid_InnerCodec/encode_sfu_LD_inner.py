#!/usr/bin/env python

# process SUF-HW dataset
#
# Warning: This script is deprecated and will be removed in future releases. Use encode_sfu.py

import os
import sys
cmd = ['python', 'encode_sfu.py', 'LD_inner'] + sys.argv[1:]
ret = os.system(' '.join(map(str,cmd)))
exit(ret)
