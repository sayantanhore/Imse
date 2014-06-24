# Create your views here.

from django.http import HttpResponse
from django.template.loader import get_template
from django.template import Context
from Intelligence.models import Image, Experiment, Iteration
import random, math, time, os
import numpy
import sys, pickle

from Intelligence.GPSOM import GPSOM
from Intelligence.GPSOMmulti import GPSOMmulti
#from Intelligence.GPUCB import ucbGP
from Intelligence.Exploitation import Exploitation
from Intelligence.Random import Random
from Intelligence.path.Path import *
from Intelligence.Util import LoadInitialImages



# Refer to path/Path.py for current working dataset

data_color = numpy.load(DATA_PATH + 'kernel-cl-'+str(IMAGENUM)+'.npy')
data = data_color
#data_texture = numpy.load(DATA_PATH + 'kernel-ht-'+str(IMAGENUM)+'.npy')
#data_shape = numpy.load(DATA_PATH + 'kernel-eh-'+str(IMAGENUM)+'.npy')

# Firstround Images
firstround_images_shown = None
clusters = None

# Predictor
predictor = None

    
# Start Search

def start_search(request):
    t = get_template('index.html')
    objs = Image.objects.all()[:IMAGENUM]
    # 2. select an image to show to the user
    target_img = random.choice(objs)
    request.session.flush()
    
    e = Experiment(
                sessionid=request.session.session_key,
                #username = request.GET['username'],
                target=target_img,
                iterations = 0,
                excellents = 0,
                goods = 0,
                satisfactories = 0,
                images_number_total = IMAGENUM,
                images_number_iteration = 12,
                finished=False,
                timestart = time.time(),
                timefinish = 0
                )
    
    e.save()
    # For apache server 
    #html = t.render(Context({'image' : '/data/Imse/media/' + target_img.filename}))
    # For django local
    #html = t.render(Context({'image' : '/media/' + target_img.filename}))
    html = t.render(Context({}))
    # For django local
    return HttpResponse(html)
    

# Target Search

def target_search(request):
    t = get_template('Intelligence/target.html')
    objs = Image.objects.all()[:IMAGENUM]
    # 2. select three images to show to the user
    
    url = request.get_full_path()
    
    sample_images_no = []
    
    if "Night" in url:
        tags = pickle.load(open(DATA_PATH + "tag_to_img_" + str(IMAGENUM)))
        #candidates = tags["Night"]
        candidates = [19824, 6929, 14122]
        sample_images_no = random.sample(candidates, 3)
        
    elif "Rose" in url:
        tags = pickle.load(open(DATA_PATH + "tag_to_img_" + str(IMAGENUM)))
        #candidates = tags["Walk"]
        candidates = [8919, 378, 574] 
        sample_images_no = random.sample(candidates, 3)
        
    elif "Building" in url:
        tags = pickle.load(open(DATA_PATH + "tag_to_img_" + str(IMAGENUM)))
        candidates = tags["Walk"]
        #candidates = [12433, 20398, 24141]
        candidates = [20398, 21299, 22280]
        sample_images_no = random.sample(candidates, 3)
    
    elif "Waterfall" in url:
        tags = pickle.load(open(DATA_PATH + "tag_to_img_" + str(IMAGENUM)))
        #candidates = tags["Waterfall"]
        candidates = [24079, 17946, 1388]
        sample_images_no = random.sample(candidates, 3)
    
    target_img = random.choice(objs)
    request.session.flush()
    
    e = Experiment(
                sessionid=request.session.session_key,
                #username = request.GET['username'],
                target_type="target",
                target=target_img,
                iterations = 0,
                excellents = 0,
                goods = 0,
                satisfactories = 0,
                images_number_total = IMAGENUM,
                images_number_iteration = 18,
                finished=False,
                timestart = time.time(),
                timefinish = 0
                )
    
    e.save()
    
    sample_images = []
    
    for im in sample_images_no:
        sample_images.append(Image.objects.get(index=im))
    
    images = []
    for s in sample_images :
            images.append({ 'image': MEDIA_PATH + s.filename,
                            'link': s.filename,
                            })
    html = t.render(Context({
                             
                             'image_list' : images,
                            }))
    return HttpResponse(html)
    
   
# Open Search

def open_search(request):
    t = get_template('Intelligence/open.html')
    objs = Image.objects.all()[:IMAGENUM]
    # 2. select an image to show to the user
    target_img = random.choice(objs)
    request.session.flush()
    
    e = Experiment(
                sessionid=request.session.session_key,
                #username = request.GET['username'],
                target_type="open",
                target=target_img,
                iterations = 0,
                excellents = 0,
                goods = 0,
                satisfactories = 0,
                images_number_total = IMAGENUM,
                images_number_iteration = 18,
                finished=False,
                timestart = time.time(),
                timefinish = 0
                )
    
    e.save()
    html = t.render(Context({}))
    return HttpResponse(html)

# Category Search

def category_search(request):
    t = get_template('Intelligence/category.html')
    objs = Image.objects.all()[:IMAGENUM]
    # 2. select an image to show to the user
    target_img = random.choice(objs)
    request.session.flush()
    
    e = Experiment(
                sessionid=request.session.session_key,
                #username = request.GET['username'],
                target_type="category",
                target=target_img,
                iterations = 0,
                excellents = 0,
                goods = 0,
                satisfactories = 0,
                images_number_total = IMAGENUM,
                images_number_iteration = 18,
                finished=False,
                timestart = time.time(),
                timefinish = 0
                )
    
    e.save()
    html = t.render(Context({}))
    return HttpResponse(html)

    
# FirstRound

def firstround_search(request):
    print "Firstround..." 
    #t = get_template('Intelligence/search.html')
    colors = request.GET['colors']
    print "Colors :: " + colors
    print request.session.session_key
    no_of_images = request.GET['no_of_images']
    no_of_images = int(no_of_images.encode("utf-8"))
    initial_image_selector = LoadInitialImages.LoadInitialImages(colors)
    print "Helooooo"
    global clusters
    images_to_show, clusters = initial_image_selector.load_images(no_of_images)
    print "Images selected"
    global firstround_images_shown
    firstround_images_shown = images_to_show
    print "In view :: " + str(images_to_show)
    return HttpResponse(str(images_to_show))

def firstround_search2(request):
	print "Username :: " + request.GET['username']
	print "Number of images :: " + request.GET['imagesnumiteration']
	print "Feedback 123 :: " + request.GET['feedback']

# Do Search

def do_search(request, state = 'nostart'):
    start_search = time.time()
    objs = Image.objects.all()[:IMAGENUM]
    try :
        e = Experiment.objects.get(sessionid=request.session.session_key)
    except :
        return bad_session(request)

    target = e.target.index

    if state == 'start':
        
        e.images_number_iteration = int(request.GET['imagesnumiteration'].encode("utf-8"))
        
        e.algorithm = request.GET['algorithm'].encode("utf-8")
        
        e.username = request.GET['username'].encode("utf-8")
        
        e.images_number_total = IMAGENUM
        
        e.category = request.GET['category'].encode("utf-8")
        
        e.iterations += 1
        
        request.session['debug'] = bool(int(request.GET.get('debug', 0)))
        
        if e.algorithm =='GP-SOM':
            global predictor
            predictor = GPSOM.GPSOM(e.images_number_iteration, e.images_number_total, firstround_images_shown, e.category)
            '''
            if e.algorithm == 'GP-UCB':
                predictor = ucbGP.GPUCB(DATA_PATH, e.images_number_iteration, e.images_number_total, e.category)
            '''
        if e.algorithm == 'Exploitation':
            predictor = Exploitation.Exploitation(e.images_number_iteration, e.images_number_total, e.category)
        if e.algorithm == 'Random':
            predictor = Random.Random(e.images_number_iteration, e.images_number_total, e.category)
        if e.algorithm == 'GP-SOM-multi':
            predictor = GPSOMmulti.GPSOMmulti(e.images_number_iteration, e.images_number_total, e.category)

        images_shown = firstround_images_shown
        
        #predictor.images_shown = images_shown
        predictor.previouse_images = images_shown
        predictor.iteration += 1

    else:
        print predictor
        #global predictor
        #predictor = request.session['calc']
    
    # Get a feedback vector using previouse_images from GPSOM and state
    feedback = []
    #marks = []
    #for im in predictor.previouse_images:
        #feedback.append(float(request.GET.get(Image.objects.get(index=im).filename,0)))
    feedback = request.GET['feedback'].encode("utf-8").replace("[", "").replace("]", "").replace("\"", "").split(",")
    feedback = [float(f) for f in feedback]
    accepted = request.GET['accepted'].encode("utf-8")
    
        #marks.append(int(request.GET.get('mark'+(Image.objects.get(index=im).filename),0)))
    #print e.algorithm
    
    #e.satisfactories += marks.count(1)
    #e.goods += marks.count(2)
    #e.excellents += marks.count(3)
    
    i = Iteration(experiment = e,
                iteration = e.iterations,
                images_shown = str(predictor.previouse_images),
                feedback = str(feedback),
                #marks = str(marks),
                time = time.time()
                  )
    
    i.save()
    
    if e.algorithm == 'GP-SOM-multi':
        ims, weightDictionary = predictor.Predict(feedback, data_color, data_texture, data_shape)

	    # Start (Added by Sayantan 24/07/13)
        
        e.colorWeight = weightDictionary["colorWeight"]
        e.textureWeight = weightDictionary["textureWeight"]
        e.shapeWeight = weightDictionary["shapeWeight"]
        e.colorPearson = weightDictionary["colorPearson"]
        e.texturePearson = weightDictionary["texturePearson"]
        e.shapePearson = weightDictionary["shapePearson"]
        e.save()
        
        # End

    else:
        
        if accepted == "true":
            ims = predictor.Predict(feedback, True)
        elif accepted == "false":
            ims = predictor.Predict(feedback, False)
        

    if request.GET.get('action')=='Finish!':
        e.finished = True
        e.timefinish = time.time()
        e.save()
        
        t = get_template('Intelligence/finished.html')
        html = t.render(Context({
                        'iterations': e.iterations
                     }))
        
        return HttpResponse(html)

    else:
        # Save GPSOM in cookies
        #request.session['calc'] = predictor
        
        '''distance = 1-(numpy.load(filename_distance))
        distances = distance[e.target.index,:]
        d = distances[ims]'''
        e.iterations += 1
        e.save()
    
        print "Hello ..." + str(ims)
    return HttpResponse(str(ims))



def do_search2(request, state='notstart'):
    
    #print 'STATE = ', request['selection']
    #print 'SESSION = ',  request.session.session_key
    
    start_search = time.time()
    objs = Image.objects.all()[:IMAGENUM]
    try :
        e = Experiment.objects.get(sessionid=request.session.session_key)
    except :
        return bad_session(request)   
    # this is the first time that do_search has been called
    # for the current session id
    target = e.target.index
    '''
    distance1 = numpy.load(filename_distance1)
    distance2 = numpy.load(filename_distance2)
    distance3 = numpy.load(filename_distance3)
    
    d1 = distance1[target,:]
    d2 = distance2[target,:]
    d3 = distance3[target,:]
    '''
    
    if state == 'start' :
        e.images_number_iteration = int(request.GET['imagesnumiteration'])
        e.algorithm = request.GET.get('algorithm')
        e.username = request.GET.get('username')
        e.images_number_total = IMAGENUM
        e.category = str(request.GET.get('category'))
        
        request.session['debug'] = bool(int(request.GET.get('debug', 0)))
        print e.algorithm
        if e.algorithm =='GP-SOM':
            predictor = GPSOM.GPSOM(e.images_number_iteration, e.images_number_total, e.category)
        '''
        if e.algorithm == 'GP-UCB':
            predictor = ucbGP.GPUCB(DATA_PATH, e.images_number_iteration, e.images_number_total, e.category)
        '''
        if e.algorithm == 'Exploitation':
            predictor = Exploitation.Exploitation(e.images_number_iteration, e.images_number_total, e.category)
        if e.algorithm == 'Random':
            predictor = Random.Random(e.images_number_iteration, e.images_number_total, e.category)
        if e.algorithm == 'GP-SOM-multi':
            predictor = GPSOMmulti.GPSOMmulti(e.images_number_iteration, e.images_number_total, e.category)
        ims = predictor.FirstRound()
        
        
        
        
         
    else:
        predictor = request.session['calc']
        # Get a feedback vector using previouse_images from GPSOM and state
        feedback = []
        marks = []
        for im in predictor.previouse_images:
            feedback.append(float(request.GET.get(Image.objects.get(index=im).filename,0)))
            #marks.append(int(request.GET.get('mark'+(Image.objects.get(index=im).filename),0)))
        #print e.algorithm
        
        #e.satisfactories += marks.count(1)
        #e.goods += marks.count(2)
        #e.excellents += marks.count(3)
        
        i = Iteration(experiment = e,
                    iteration = e.iterations,
                    images_shown = str(predictor.previouse_images),
                    feedback = str(feedback),
                    #marks = str(marks),
                    time = time.time()
                      )
        i.save()
        
        if e.algorithm == 'GP-SOM-multi':
            ims, weightDictionary = predictor.Predict(feedback, data_color, data_texture, data_shape)

	    # Start (Added by Sayantan 24/07/13)
            
            e.colorWeight = weightDictionary["colorWeight"]
            e.textureWeight = weightDictionary["textureWeight"]
            e.shapeWeight = weightDictionary["shapeWeight"]
            e.colorPearson = weightDictionary["colorPearson"]
            e.texturePearson = weightDictionary["texturePearson"]
            e.shapePearson = weightDictionary["shapePearson"]
            e.save()
            
            # End

        else:
            ims = predictor.Predict(feedback, data)
      
    # Get objects by indeces predicted by GPSOM    
    if request.GET.get('action')=='Finish!':
        e.finished = True
        e.timefinish = time.time()
        e.save()
        
        t = get_template('Intelligence/finished.html')
        html = t.render(Context({
                        'iterations': e.iterations
                     }))
        
        return HttpResponse(html)
    else:
        samp = []
        for im in ims:
            samp.append(Image.objects.get(index=im))
    
        # Save GPSOM in cookies
        request.session['calc'] = predictor
        
        '''distance = 1-(numpy.load(filename_distance))
        distances = distance[e.target.index,:]
        d = distances[ims]'''
        e.iterations += 1
        e.save()
    
        images = []
        for s in samp :
            images.append({ 'image': MEDIA_PATH + s.filename,
                            'link': s.filename,
                            'finish' : "/imse_dev/finish/%s/" % s.filename,
                            'distance' : []
                            })
    
    
        t = get_template('Intelligence/search.html')
        html = t.render(Context({
                            'selection' : request.session.get('selection',''),
                            'image_list' : images, 
                            'debug' : request.session['debug'], 
                            'random' : 1, #int(alg == 'random'),  Don't know what this 'random' is
                            'target' : MEDIA_PATH + e.target.filename
                        }))
    return HttpResponse(html)
