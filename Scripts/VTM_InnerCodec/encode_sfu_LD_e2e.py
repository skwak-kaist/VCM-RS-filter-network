#!/usr/bin/env python

# process SUF-HW dataset using VTM as the inner codec
# number of tasks 84
#
# Usage of the script: 
#   <script_name> <task_id>
# This script may be useful to process the data on a cluster using CPUs. 
# The script process one item identified by the task_id. The task id is from
# 1 to the total number of tasks. 
#
# Warning: This script is deprecated and will be removed in future releases. Use encode_sfu.py

import os
import sys
cmd = ['python', 'encode_sfu.py', sys.argv[1], 'LD_e2e'] + sys.argv[2:]
ret = os.system(' '.join(map(str,cmd)))
exit(ret)
