#!/usr/bin/env python

# process TVD dataset
#
# Warning: This script is deprecated and will be removed in future releases. Use encode_tvd_tracking.py

import os
import sys
cmd = ['python', 'encode_tvd_tracking.py', 'RA_inner'] + sys.argv[1:]
ret = os.system(' '.join(map(str,cmd)))
exit(ret)
