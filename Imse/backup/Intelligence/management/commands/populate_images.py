from django.core.management.base import BaseCommand, CommandError
from Intelligence.models import Image
import os

from xml.dom.minidom import parse


class Command(BaseCommand) :
    args = '<directory containing features and files>'
    help = 'populates database with content of a directory'    
    
    def handle(self, *args, **options) :
        if len(args) != 1 :
            raise CommandError("populate command needs dir as argument")

        no_of_images = args[0]
        
        namesfile = '/data/Imse/Data/imageset' + no_of_images + '.nm'
        #namesfile = '/home/fs/konyushk/fs/konyushk/Imse/' + path + ''
        
        
        #kernel = numpy.load(kernelfile)
        #features = numpy.loadtxt(featuresfile)
        names = [line.strip() for line in open(namesfile).readlines()]
        
        images_number = len(names)
        
        for pic in range(images_number):
            a = Image()
            print pic
            a.index = pic
            a.filename = names[pic]
            a.save()