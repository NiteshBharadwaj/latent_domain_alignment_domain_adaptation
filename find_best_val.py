import sys
import os
filename = sys.argv[1]
filename = './record_pacs/'+filename+'/pacs_0_val.txt'
max_acc = -1
best_epoch = -1
with open(filename,'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        max_acc = max(max_acc, float(line))
        best_epoch = idx
print('Maximum Accuracy', max_acc)
print('Best epoch', best_epoch)
