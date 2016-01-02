from django.conf.urls import patterns, include, url
from Intelligence.views import start_search, do_search, firstround_search
from django.conf import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()
urlpatterns = patterns('',
    	# Examples:
	# url(r'^$', 'Imse.views.home', name='home'),
	# url(r'^Imse/', include('Imse.foo.urls')),

	# Uncomment the admin/doc line below to enable admin documentation:
	# url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

	# Uncomment the next line to enable the admin:
	# url(r'^admin/', include(admin.site.urls)),

    url(r'^data/$', 'django.views.static.serve', {'document_root': settings.DATA_PATH}),
    url(r'^start/media/(.*)$', 'django.views.static.serve', {'document_root': settings.MEDIA_PATH}),
    url(r'^media/(.*)$', 'django.views.static.serve', {'document_root': settings.MEDIA_PATH}),

    url(r'^start/$', start_search),
    url(r'^firstround/$', firstround_search),
    url(r'^predict/(.*)/$', do_search),
    url(r'^predict/$', do_search),
)
