from django.core.management.base import BaseCommand, CommandError
from Intelligence.models import Image

from xml.dom.minidom import parse
import glob, os, math, copy
import numpy
from scipy import spatial
import pickle
import scipy

import sys
import cv

b = 8


def rgb_histogram(src):
    B_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
    G_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
    R_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
    cv.Split(src, B_plane, G_plane, R_plane, None)
    planes = [B_plane, G_plane, R_plane]
    B_bins = b
    G_bins = b
    R_bins = b
    bins = [B_bins, G_bins, R_bins]
    B_ranges = [0, 225]
    G_ranges = [0, 225]
    R_ranges = [0, 225]
    ranges = [B_ranges, G_ranges, R_ranges]
    hist = cv.CreateHist(bins, cv.CV_HIST_ARRAY, ranges, 1)
    cv.CalcHist([cv.GetImage(i) for i in planes], hist)
    cv.NormalizeHist(hist, 1)
    return hist


class Command(BaseCommand):
    args = '<number of images>'
    help = 'homogeneous texture distance'

    def handle(self, *args, **options):
        if len(args) != 1:
            raise CommandError("Kernel command needs a number of images in the dataset")

        images_number = int(args[0])

        cl_data = numpy.load('/data/Imse/Data/cl' + str(images_number) + '.npy')
        cl_std = numpy.std(cl_data, 0)
        distances = scipy.spatial.distance.cdist(cl_data, cl_data, 'cityblock') / len(cl_std)
        numpy.save('/data/Imse/Data/cl_distances' + str(images_number), distances)
