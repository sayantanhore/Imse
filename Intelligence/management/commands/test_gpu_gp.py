from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import Intelligence.GP_GPU.gp_cuda as gp_cuda

import numpy as np
import sys
import time


def test_gpu_gp(data, test_input_path):
    feedback = np.load(test_input_path + '0_feedback.npy')
    feedback_indices = np.load(test_input_path + '0_feedback_indices.npy')
    K_diag_noise = np.load(test_input_path + '0_random_K.npy')
    K_xx_noise = np.load(test_input_path + '0_random_K_xx.npy')
    mean, variance = gp_cuda.gaussian_process(data, feedback, feedback_indices, K_noise=K_diag_noise, K_xx_noise=K_xx_noise, debug=True)


class Command(BaseCommand):
    def handle(self, *args, **options):
        data = np.asfarray(np.load(settings.DATA_PATH + "cl25000.npy"), dtype="float32")
        input_path = settings.DATA_PATH + 'speedtest_input/'
        #print('sys.argv[1] == test')
        print('Number of image feature vectors: ' + str(np.shape(data)[0]))
        test_gpu_gp(data, input_path)
        exit()
