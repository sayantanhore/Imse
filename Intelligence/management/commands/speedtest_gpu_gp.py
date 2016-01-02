from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import Intelligence.GP_GPU.gp_cuda as gp_cuda

import numpy as np
import sys
import time


def test_gpu_gp(data, test_input_path):
    totaltime = 0
    feedback = []
    feedback_indices = []
    for i in range(63):
        new_input = float(np.random.randint(11)) / 10
        new_input_idx = np.random.randint(np.shape(data)[0])
        feedback.append(new_input)
        feedback_indices.append(new_input_idx)
        t = time.time()
        mean, variance = gp_cuda.gaussian_process(data, feedback, feedback_indices)
        itertime = time.time() - t
        print(str(i) + '\t' + str(itertime) + '\t' + str(len(feedback)))
        totaltime += itertime
    print('Total time:', str(totaltime))


class Command(BaseCommand):
    def handle(self, *args, **options):
        data = np.asfarray(np.load(settings.DATA_PATH + "0_50k_feat.npy"), dtype="float32")
        if len(args) > 0:
            n = int(args[0])
            if n > 1:
                for i in range(1, n):
                    tmpdata = np.load(settings.DATA_PATH + str(i) + '_50k_feat.npy')
                    data = np.concatenate((data, tmpdata))
        input_path = settings.DATA_PATH + 'speedtest_input/'
        #print('sys.argv[1] == test')
        print('Number of image feature vectors: ' + str(np.shape(data)[0]))
        test_gpu_gp(data, input_path)
        exit()
