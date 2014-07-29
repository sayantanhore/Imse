#!/usr/bin/env python

import sys
import numpy as np


if __name__ == '__main__':
    gpu_filename = sys.argv[1]
    cpu_filename = sys.argv[2]

    gpu_data = np.load(gpu_filename)
    cpu_data = np.load(cpu_filename)
    print(gpu_data)
    print(cpu_data)

    all_close = np.allclose(gpu_data, cpu_data)
    print('All values close to each other:', all_close)
    if not all_close:
        print('Number of values not close to each other:', sum(np.isclose(gpu_data, cpu_data)))
        print('Total number of values:', len(gpu_data))
