import sys
import json
import ijson
import re
from collections import defaultdict, OrderedDict
from mpi4py import MPI

### --- MPI code. --- ###
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Read input file from STDIN.
if rank == 0:
    json_file = open(sys.argv[1], 'r')
    print(json_file.read())
else:
    json_file = None

# Scatter data to other processes.
json_part = comm.scatter(json_file, root = 0)

# Each process tries to open json and fix if necessary.
try:
    json_dict = json.loads(json_part)
    print('JSON data parsing completed without errors.')
except:
    fixed_json = json_part.read()[:-2] + ']}'
    json_dict = json.loads(fixed_json)

for row in json_dict['rows']:
    print('Process %d of %d printing: ' % (rank, size) + row['doc']['text'])




# Merge result.
