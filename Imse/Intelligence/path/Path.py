# The loaded data files are the following:
#data_color = numpy.load(DATA_PATH + 'kernel-cl-'+str(IMAGENUM)+'.npy')
#data = data_color

# Loading 25000 dataset
IMAGENUM = 25000

import socket


if socket.gethostname() == 'iitti':
    DATA_PATH = '/home/lassetyr/programming/Imse/data/Data/'
    base_path = '/home/lassetyr/programming/Imse/Imse/'
    MEDIA_PATH = '/home/lassetyr/programming/Imse/data/media/'
else:
    DATA_PATH = "/ldata/IMSE/data/Data/"
    base_path = '/ldata/IMSE/Imse/Imse/'
    MEDIA_PATH = '/ldata/IMSE/data/media/'
# Loading 10000 dataset
#IMAGENUM = 10000
#DATA_PATH = '/ldata/IMSE/data/Data10000/'

# For Apache2-mod_wsgi Server
#MEDIA_PATH = '/imse_dev2/media/'

# For Django Local (for 25k images)


# For accessing modules from Django dev server

FILE_ROOT_PATH = '/home/lassetyr/programming/Imse/Imse/'
