from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import Intelligence.GP_CPU.gp as gp

import numpy as np
import sys
import time


def test_cpu_gp(data, test_input_path):
    totaltime = 0
    for i in range(3):
        inputprefix = test_input_path + str(i)
        feedback = np.load(str(inputprefix) + '_feedback.npy')
        feedback_indices = np.load(str(inputprefix) + '_feedback_indices.npy')
        gpob = gp.GP()
        t = time.time()
        mean, var = gpob.GP(feedback_indices, feedback, data)
        itertime = time.time() - t
        print(str(i) + '\t' + str(itertime) + '\t' + str(len(feedback)))
        totaltime += itertime
    print('Total time:', str(totaltime))


class Command(BaseCommand):
    def handle(self, *args, **options):
        covariance_matrix = np.asfarray(np.load(settings.DATA_PATH + 'cl_distances25000.npy'), dtype='float32')
        input_path = settings.DATA_PATH + 'speedtest_input/'
        #print('sys.argv[1] == test')
        test_cpu_gp(covariance_matrix, input_path)
        exit()
