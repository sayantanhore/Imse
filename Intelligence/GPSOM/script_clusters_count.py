import testrun as tr
import numpy as np

DATA_PATH = "/data/Imse/Data/"
IMAGENUM = 25000

data_color = np.load(DATA_PATH+ 'kernel-cl-'+str(IMAGENUM)+'.npy')
#data_texture = np.load(DATA_PATH+ 'kernel-ht-'+str(IMAGENUM)+'.npy')
#data_shape = np.load(DATA_PATH+ 'kernel-eh-'+str(IMAGENUM)+'.npy')

for i in range(2):
	#tr.TestRun(i, 15, data_color, data_texture, data_shape)
	tr.TestRun(i, 5, data_color)

del data_color
#del data_texture
#del data_shape