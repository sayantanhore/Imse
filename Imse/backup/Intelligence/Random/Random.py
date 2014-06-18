import pickle
import numpy
import math
import random

class Random(object):
    
    def __init__(self, images_number_iteration, images_number_total, category):
        self.setsize = images_number_iteration
        self.images_number = images_number_total
        self.category = category
        self.feedback = []
        #self.images_shown = []
        self.previouse_images = []
        self.selected_images = []
    
    def FirstRound(self):
        '''
        all_images = numpy.arange(0,self.images_number-1)
        numpy.random.shuffle(all_images)
        images = []
        i = 0
        while len(images)<self.setsize:
            if all_images[i] not in self.previouse_images:
                if all_images[i] not in self.selected_images:
                    images.append(all_images[i])
            i += 1
        '''
        
        all_images = numpy.arange(0, self.images_number)
        #if(self.category == "None"):
        images_per_group = int(math.ceil(self.images_number / self.setsize))
        #else:
            #images_per_group = 2 * int(math.ceil(self.images_number / self.setsize))
        
        images = []
        
        image_counter = 0
        while(image_counter < self.images_number -1):
            if((self.images_number - image_counter) >= images_per_group) and ((self.images_number - image_counter) < 2 * images_per_group):
                images_group = all_images[image_counter:]
                image_counter = self.images_number
            else:
                images_group = all_images[image_counter:image_counter + images_per_group]
                image_counter += images_per_group
            
            numpy.random.shuffle(images_group)
            image = images_group[0]
            images.append(image)
        '''
        numpy.random.shuffle(all_images)
        images = []
        for i in all_images[:self.setsize]:
            images.append(i)
        '''
        '''
        if(self.category != "None"):
            tags = pickle.load(open("/data/Imse/Data/tag_to_img_" + str(self.images_number)))
            candidates = tags[self.category]
            images_from_selected_category = random.sample(candidates, self.setsize / 2)
            images.extend(images_from_selected_category)
        '''
        random.shuffle(images)
        
        
        #self.images_shown = images
        
        self.previouse_images = images
        return images
    
    def Predict(self, feedback, data):
        
        self.feedback = self.feedback + feedback
        i = 0
        for f in feedback:
            #if f!=0:
            self.selected_images.append(self.previouse_images[i])
            i += 1  
        
        all_images = numpy.arange(0, self.images_number)
        images_per_group = int(math.ceil(self.images_number / self.setsize))
        
        images = []
        image_counter = 0
        
        while(image_counter < self.images_number -1):
            if((self.images_number - image_counter) >= images_per_group) and ((self.images_number - image_counter) < 2 * images_per_group):
                images_group = all_images[image_counter:]
                image_counter = self.images_number
            else:
                images_group = all_images[image_counter:image_counter + images_per_group]
                image_counter += images_per_group
            
            numpy.random.shuffle(images_group)
            
            for j in range(len(images_group)):
                if images_group[j] not in self.previouse_images and images_group[j] not in self.selected_images:
                    images.append(images_group[j])
                    break
        self.previouse_images = images
        return images