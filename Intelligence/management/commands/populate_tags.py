from django.core.management.base import BaseCommand, CommandError
from Imse.Intelligence.models import Image
import os
import pickle
import json
import re


from xml.dom.minidom import parse


class Command(BaseCommand) :
    args = '<directory containing features and files>'
    help = 'populates database with content of a directory'    
    
    def handle(self, *args, **options) :
        if len(args) != 1 :
            raise CommandError("populate command needs dir as argument")

        no_of_images = args[0]
        
        tag_list = []
        tag_to_img = dict()
        img_to_tag = dict()
        
        
        for i in range(int(no_of_images)):
            filename = "/data/Imse/media/meta/tags/tags" + str(i+1) + ".txt"
            file = open(filename, "r")
            lst = []
            for line in file:
                lst.append(line.rstrip('\r\n').title())
                tag_list = tag_list + lst
            file.close()
            if(len(lst) != 0):
                for tag in lst:
                    img_to_tag.setdefault(i, []).append(tag.title())
            else:
                img_to_tag.setdefault(i, []).append("None")

	pickle.dump(img_to_tag, open("/data/Imse/Data/img_to_tag_" + str(no_of_images), "w"))	
        
        tag_list = list(set(tag_list))
        
        for tag in tag_list:
            for i in range(len(img_to_tag)):
                if(len(img_to_tag[i]) != 0):
                    if(tag in img_to_tag[i]):
                        tag_to_img.setdefault(tag, []).append(i)
                        
        pickle.dump(tag_to_img, open("/data/Imse/Data/tag_to_img_" + str(no_of_images), "w"))
        
        # Creating json file for tags
        
        tag_to_img = pickle.load(open("/data/Imse/Data/tag_to_img_" + str(no_of_images)))
        
        tags_to_show = []
        
        for tag in tag_to_img:
            #if(tag.isdigit() == False and re.match('^[\w-]+$', tag) and len(tag_to_img[tag]) > 9):
            if(tag.isdigit() == False and re.match('^[A-Za-z_-]+$', tag) and len(tag_to_img[tag]) > 9):
                if(not tag in open("/data/Imse/Data/places.txt").read()):
                    tags_to_show.append(tag)
        tags_to_show.sort()
        
        tags_to_show = ["None"] + tags_to_show        
        json_file = open("/opt/Imse/Imse/static/scripts/tags.json","w")
        json.dump(tags_to_show, json_file)
        json_file.close()
        
        return 
