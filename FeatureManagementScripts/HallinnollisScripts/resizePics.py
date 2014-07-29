import os
import sys

fromfolder = '/home/IMSE/data/media/'
tofolder = '/home/overfeat/cropix/'

for f in os.listdir(fromfolder):
	print "Processing %s." %(f)
	os.system('convert %s%s -resize 231x231^ -gravity center -extent 231x231 %scropd%s' %(fromfolder, f, tofolder,f))
	


