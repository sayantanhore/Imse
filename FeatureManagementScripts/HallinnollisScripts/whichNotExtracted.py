#in the features folder there were missing entries due to poor extractFeatures.py script.
#I'm finding out which feature vectores were missing with this script by
#looking at the filenames


import sys
import os
import re

folder = 'features/'
filesused = dict()
allfiles = range(25001)
for file in os.listdir(folder):
	val = re.findall(r'\d+', file)[0]
	print val
	filesused[val] = 'asd'

for file in allfiles:
	if str(file) not in filesused.keys():
        	print file
'''
0
696
791
898
1926
1944
2250
3917
3919
3935
4914
4953
4971
5927
5935
5936
6935
6940
7904
7921
8932
8981
9934
9973
9996
19120
19318
19648
19668
19717
19738
19861
22032
22061
22092
22559
'''
