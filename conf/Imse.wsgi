"""
WSGI config for Imse project.

This module contains the WSGI application used by Django's development server
and any production WSGI deployments. It should expose a module-level variable
named ``application``. Django's ``runserver`` and ``runfcgi`` commands discover
this application via the ``WSGI_APPLICATION`` setting.

Usually you will have the standard Django WSGI application here, but it also
might make sense to replace the whole Django WSGI application with a custom one
that later delegates to the Django one. For example, you could introduce WSGI
middleware here, or combine a Django application with an application of another
framework.

"""



import os
import sys
sys.path.append('/ldata/Imse/conf/')

# We defer to a DJANGO_SETTINGS_MODULE already in the environment. This breaks
# if running multiple sites in the same mod_wsgi process. To fix this, use
# mod_wsgi daemon mode with each site in its own daemon process, or use


os.environ["DJANGO_SETTINGS_MODULE"] = "settings"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-7.5/lib64"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
os.environ['PATH'] += os.pathsep + '/home/fs/hore/anaconda2/envs/image_search/bin' + os.pathsep + '/usr/local/cuda-7.5/bin'
os.environ['PYTHON_PATH_IN_USE'] = '/home/fs/hore/anaconda2/envs/image_search/bin/python'
print("Env loaded")


from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
print("Application loaded")



# This application object is used by any WSGI server configured to use this
# file. This includes Django's development server, if the WSGI_APPLICATION
# setting points here.
#from django.core.wsgi import get_wsgi_application
#application = get_wsgi_application()
# Apply WSGI middleware here.
# from helloworld.wsgi import HelloWorldApplication
# application = HelloWorldApplication(application)
