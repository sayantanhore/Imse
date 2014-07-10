import os
import sys

fromfolder = '/home/IMSE/data/media/'
tofolder = '/home/overfeat/cropix/'

for i in range(3,10):
	print "Processing %s." %(i)
	for j in range(1,4): #do subfolders imgs3/10 imgs3/20 and imgs3/30,
		os.system('mkdir /home/overfeat/cropix/imgs%d/%d0' %(i, j))
	for j in range(1,4): #into which goes imgs30* imgs31* imgs32* into the first and the next three into next and next folders
		os.system('mv /home/overfeat/cropix/imgs%d/cropdim%d%d* /home/overfeat/cropix/imgs%d/10/ ' %(i,i,j-1,i))
		os.system('mv /home/overfeat/cropix/imgs%d/cropdim%d%d* /home/overfeat/cropix/imgs%d/20/ ' %(i,i,j+2,i))
		os.system('mv /home/overfeat/cropix/imgs%d/cropdim%d%d* /home/overfeat/cropix/imgs%d/30/ ' %(i,i,j+5,i))
