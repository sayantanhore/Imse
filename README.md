Imse
====

------------------------------------------------------------------------------------------------------------
PLEASE DO NOT CHANGE ANYTHING IN THIS BRANCH, MAKE YOUR OWN BRANCH AND CHANGE THERE ACCORDING TO YOUR NEEDS.
------------------------------------------------------------------------------------------------------------


Imse - A reinforcement learning based Image Retrieval system

This is a work in progress. This branch holds the basic version of Gaussian Process that runs on the GPU.

Refer to "localsettings.py" file for system configurations. The variables declared there are self explanatory. You can use them everywhere. Declare constant and static variables only in this file. This file is imported into settings.py and settings.py is imported elsewhere. Therefore anything declared in this file is available everywhere.

Keep all data files like "features" or "distances" etc. in "/ldata/IMSE_OLD/data/"

The "templates" folder has the html files. We start with "index.html" fired from "start_search()" in "Intelligence/views.py". You can put your own html files into "templates" and change in "views.py:start_search()".

URLs are defined in "Imse/urls.py". We hit "/firstround/" from "index.html" to fetch the initial set of images. Next iteration onwards we hit "/search/". Onclick "Done" we hit "/search/" again with parameter "finished = 'true'".

---------
THANK YOU
----------
