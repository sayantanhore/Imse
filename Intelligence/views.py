
from django.http import HttpResponse, JsonResponse
from django.template.loader import get_template
from django.template import Context
from Intelligence.models import Image, Experiment, Iteration
import random, math, time, os
import numpy
import sys, pickle

from Intelligence.GP_GPU import GP_GPU
#from Intelligence.Util import LoadInitialImages
from subprocess import Popen
from signal import SIGTERM
from django.conf import settings
import xmlrpclib
from datetime import datetime as dt
import csv
import json
from django.conf import settings

# Globals
#--------------------------------------------------------------------------------------------------------------------

results_file = None

# Subprocess handler
p = None


# Firstround Images
firstround_images_shown = None
clusters = None

# Predictor
predictor = None

# Get current time in miliseconds
current_time = lambda: int(round(time.time() * 1000))


#--------------------------------------------------------------------------------------------------------------------

# Return data file name
def get_datafile_name(username):
	timestamp = ''
	timestamp = timestamp + str(dt.now().date().year) + '_'
	timestamp = timestamp + str(dt.now().date().month) + '_'
	timestamp = timestamp + str(dt.now().date().day) + '_'

	timestamp = timestamp + str(dt.now().time().hour) + '_'
	timestamp = timestamp + str(dt.now().time().minute) + '_'
	timestamp = timestamp + str(dt.now().time().second) + '_'

	data_file = username + '_'  + timestamp + 'data.csv'
	return data_file

# Allow remote development
def enable_remote_access_control(res):


    res["Access-Control-Allow-Origin"] = "*"
    res["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    res["Access-Control-Max-Age"] = "1000"
    res["Access-Control-Allow-Headers"] = "*"
    return res

# Start Search

def start_search(request):
    print("Enter >> 'start_search'")
    print("----------------------------")
    t = get_template('index.html')
    #request.session.flush()
    '''
    e = Experiment(
                sessionid=request.session.session_key,
                #username = request.GET['username'],
                iterations = 0,
                excellents = 0,
                goods = 0,
                satisfactories = 0,
                images_number_total = settings.IMAGENUM,
                no_of_images_to_show = 12,
                finished=False,
                timestart = time.time(),
                timefinish = 0
                )
    e.save()
    '''
    html = t.render(Context({}))
    response = HttpResponse(html)
    response = enable_remote_access_control(response)
    print("Exit << 'start_search'")
    print("----------------------------")
    return response

# Initiate handles

def initiate_handles():
    # Initiate Gaussian Process on GPU
    global p
    if p == None:
	python_exec = os.environ['PYTHON_PATH_IN_USE']
    	p = Popen([python_exec, settings.FILE_ROOT_PATH + "/Intelligence/GP_GPU/gp_cuda.py", settings.DATA_PATH], env=os.environ.copy())

# FirstRound

def firstround_search(request):
    print("Enter >> 'firstround_search'")
    print("------------------------------------------------------------------------------------------------------------------------")

    if 'colors' in request.GET:
        colors = request.GET.getlist('colors')
        print("Selected Colors :: " + str(colors))
    else:
        colors = []

    no_of_images_to_show = int(request.GET['no_of_images'].encode("utf-8"))

    #initial_image_selector = LoadInitialImages.LoadInitialImages(colors)

    global clusters
    #images_to_show, clusters = initial_image_selector.load_images(no_of_images)
    #images_to_show = [18795, 10074, 11398, 5858, 19078, 24296, 2162, 17118, 24075, 21005, 16737, 24268]
    images_to_show = [2095, 24003, 4495, 3536, 3280, 19299, 84, 15335, 13063, 14562, 13189, 4892]
    #images_to_show = [2095, 24003, 4495, 3536, 3280, 19293, 84, 15335]

    global firstround_images_shown
    firstround_images_shown = images_to_show
    print("Initial images :: " + str(images_to_show))

    print("Call Handle")
    # Initiating handles
    initiate_handles()
    print("Initiated")
    print("Exit << 'firstround_search'")
    print("----------------------------")
    #return HttpResponse(str(images_to_show))
    return JsonResponse({'results': images_to_show})

# Do Search

def do_search(request, state = 'nostart'):
    print("Enter >> 'do_search'")
    print("----------------------------")
    # Fetch session
    '''
    try :
        e = Experiment.objects.get(sessionid=request.session.session_key)
    except :
        return bad_session(request)
    '''
    # Fetch request parameters
    finished = request.GET['finished'].encode("utf-8")
    print("Finished :: " + str(finished))


    # Check for finished

    if finished == "true":
	print("Finishing ...")
	'''
        e.finished = True
        e.timefinish = time.time()
        e.save()
	'''
	# Terminating GPU service
        global p
        p.send_signal(SIGTERM)
	p = None

	global predictor
	predictor = None

	print("Finished, exiting ...")
    	print("Exit << 'do_search'")
        print("-------------------------------------------------------------------------------------------------------------------------")
	return JsonResponse({'results': []})

    else:
	no_of_images_to_show = int(request.GET['no_of_images'].encode("utf-8"))
        feedback = request.GET.getlist('feedback[]')
        feedback = [float(f) for f in feedback]
        non_zero_feedback_loc = numpy.nonzero(feedback)[0]

	feedback_indices = request.GET.getlist('feedback_indices[]')
        feedback_indices = [int(f) for f in feedback_indices]
        print("Incoming feedback :: " + str(feedback))
        print("Images those received feedback :: " + str(feedback_indices))


        global predictor
        if predictor == None:
		print("Creating predictor")
		# Initialize GP
		predictor = GP_GPU.GP_GPU(no_of_images_to_show, settings.IMAGENUM, firstround_images_shown)
        	images_shown = firstround_images_shown
        	predictor.previous_images = images_shown

	# Call for prediction
	ims = []
	print("Calling predict to 'FETCH' images")
	start_time = current_time()
	ims = predictor.predict(feedback, feedback_indices, no_of_images_to_show, p)
	end_time = current_time()

   	print("Checking ims type :: " + str(ims))

        # Save iterations
	'''
	i = Iteration(experiment = e,
		iteration = e.iterations,
		images_shown = str(predictor.previous_images),
		feedback = str(feedback),
		#marks = str(marks),
		time = time.time()
	)
	i.save()
        if num_predictions != 1:
            e.iterations += 1

	# Save experiment
        e.save()
	'''
	#print("Data saved")
    	print("Exit << 'do_search'")
        print("----------------------------")
    	# Return images
	print("Returning 'FETCH'ed images ...")
	return JsonResponse({'results': ims.tolist()})


