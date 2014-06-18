from django.db import models

# Create your models here.

LEN = 128
FEATURELEN = 43
KERNELLEN = 1000

# Create Model for Images

class Image(models.Model):
    index = models.PositiveIntegerField()
    filename = models.CharField(max_length=LEN)

    def __unicode__(self):
        return u'Image: %s %d' % (self.filename, self.index)

        
# Create Model for Experiments

class Experiment(models.Model) :
    sessionid = models.CharField(max_length=LEN)
    username = models.CharField(max_length=LEN, null=True)
    algorithm = models.CharField(max_length=LEN)
    target_type = models.CharField(max_length=LEN, null=True)
    category = models.CharField(max_length=LEN, null=True)
    target = models.ForeignKey(Image, null=True)
    iterations = models.PositiveIntegerField()
    excellents = models.PositiveIntegerField(null=True)
    goods = models.PositiveIntegerField(null=True)
    satisfactories = models.PositiveIntegerField(null=True)
    images_number_total = models.PositiveIntegerField()
    images_number_iteration = models.PositiveIntegerField()
    finished = models.BooleanField()
    timestart = models.IntegerField()
    timefinish = models.IntegerField()
    
    def __unicode__(self) :
        return u'Experiment: %s %s' % (self.sessionid, self.target.filename)
        
        
# Create Model for Iterations

class Iteration(models.Model):
    experiment = models.ForeignKey(Experiment)
    iteration = models.PositiveIntegerField()
    images_shown = models.CharField(max_length=LEN)
    feedback = models.CharField(max_length=LEN)
    marks = models.CharField(max_length=LEN, null=True)
    time = models.IntegerField()