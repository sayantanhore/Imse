import os
import sys

i = sys.argv[1] #this X refers to cropix/imgsX/
#rootfolder = '/home/overfeat/cropix/imgs%s' %(i)
rootfolder = '/home/overfeat/cropix/imgsMISS'
outfold = '/home/overfeat/features'
'''
print "Processing %s." %(i)
for root, subfolders, files in os.walk(rootfolder):
	for folder in subfolders:
		print "folder %s" %(folder)
		os.system('./bin/linux_64/overfeat_batch -i %s/%s/ -o %s' %(rootfolder, folder,outfold))
'''
print "root"
os.system('./bin/linux_64/overfeat_batch -i %s/ -o %s' %(rootfolder,outfold))	


