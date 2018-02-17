#! /usr/bin/env python

# Dear Anon,
# 
# I'm not going to rewrite all of this to make it friendly to the public,
# but you should get the idea of what to do. There are a few dead ends and
# some idioms which have become pointless over development. Just calm down.
# The Dataprep class is jar of hackjob ways to getting normalized images into
# either a 'Positive' or 'Neutral' folder and then splitting off images for
# validation. SOFTrainer is where the magic happens.
#
# Basically, there's no big red button for you here, but the code should be
# instructive. Remember:
# Normalize your god damn images 224,224 (AlignDlib.py makes it easy)
# You need keras_vggface
# Nothing here is fine-tuned or perfected. Use some trial and error.
# No, I will not 'write it in C lol'
#
# I've included a trained model, which has about 95% validation accuracy
# and I've found catches about 1/4 of celebrity pedo I test on. So far,
# no obvious false positives. It's only trained on fucking white males,
# though. Other groups will come later.
# I include this because you assholes might actually do some good with it. 
#
# Basic usage for prediction would look something like:
# some_picture = numpy_array_of_picture*1./255
# vgg_face_instance = VGGFace(include_top=False,pooling='avg',input_shape=(224,224,3))
# features = vgg_face_instance.predict(some_picture[None,:,:,:])
# prediction = SOFC_model.predict(features)
# if prediction.argmax() == 1:
#	totally a pedo
# else:
#	maybe not pedo
#
# God speed.

################################################
#
# Transfer-learning based NN. Extracts features
# from VGGFace and uses those outputs to train a small
# MLP to predict a predisposition toward pedophilia.
#
################################################

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras_vggface.vggface import VGGFace
from keras.utils.np_utils import to_categorical
import math
import matplotlib.pyplot as plt
import logging
import pdb    
import sqlite3
import json
from random import shuffle
from math import ceil

logging.basicConfig(filename='SOCerror.log',level=logging.INFO)

### GLOBAL VARIABLES ###
validation_split = 0.25
positive_directory = 'Data/Offender/Train/Positive/'
neutral_directory = 'Data/Offender/Train/Neutral/'
fitw_directory = 'Source Data/Normalized Faces/Faces in the Wild/'
control_mugs_directory = 'Source Data/Normalized Faces/Normalized Inmates/'
gnames_directory = 'Source Data/Normalized Faces/Normalized Google/'
us10k = 'Source Data/Normalized Faces/10k Faces/'
race_directory = 'Data/Race/Train/'
demographic_group = 'white male'
off_sex = '"M"'
off_race = '"W"'
victim_group = ['"child"','"minor"']
victim_sex = ['"M"','"F"','"UNK"']
crime_category = ['"assault"', '"indecency"']


class DataPrep:
    """
    The DataPrep class is meant to take our images which have
    already been normalized with DLIB to 244x244 images centered
    on detected faces. There are four major tasks:

    - move sex offenders to Data/Offender/Positive
    - move a sample of the Faces in the Wild dataset to Neutral
    - move a sample of faces used from the racial classifier to Neutral
    - move a sample of non-sex offender mugshots to Neutral

    The objective is to limit the effect of biases within these samples
    by included a wide range of sampling. At the present, access to
    a large dataset of truly random faces is not something that I have
    access to. Also, statistically, there is a high likelihood that some
    number of pedophiles are within the 'neutral' datasets. Controlling
    for this problem has not, as of yet, been solved. Consequently, the
    results will suffer from this statistical noise.

    The class variables act as a control panel to allow for modification
    of criteria for selecting positive and neutral faces.
    """
      
    def __init__(self):
        self.sofdb = 'SexOffender.db'
        self.imgdir = 'Source Data/Normalized Faces/Normalized SOF/'
        self.pdir = positive_directory
        self.ndir = neutral_directory
        self.fitw = fitw_directory
        self.ctl = control_mugs_directory
        self.race = race_directory
        self.goog = gnames_directory
        self.dem = demographic_group
        self.osex = off_sex
        self.orace = off_race
        self.us10k = us10k
        self.vgroup = victim_group
        self.vsex = victim_sex
        self.cat = crime_category
        self.val = validation_split
        self.sample = 0.5 # percentage of name-search results to use

    def query_offenders(self):
        # Construct a SQLite query based on variables above. Return query object
        conn = sqlite3.connect(self.sofdb)
        cursor = conn.cursor()
        base_query = 'SELECT people.id FROM people '
        event_join = 'INNER JOIN event ON event.pid = people.id '
        crime_join = 'INNER JOIN crime ON crime.id = event.cid '
        where = "WHERE "
        offprofile = '(people.sex = {0} AND people.race = {1}) AND '.format(
            self.osex,self.orace)
        vgroup = ''
        for i in range(len(self.vgroup)):
            vgroup += 'crime.v_age_group = {0} OR '.format(self.vgroup[i])
        # Strip trailing 'OR' and add a space
        vgroup = vgroup.strip('OR ')
        if len(vgroup) > 1:
            vgroup = '('+vgroup+')'
            vgroup += ' AND '
        vsex = ''
        for i in range(len(self.vsex)):
            vsex += 'crime.v_sex = {0} OR '.format(self.vsex[i])
        vsex = vsex.strip('OR ')
        if len(vsex) > 1:
            vsex = '('+vsex+')'
            vsex += ' AND '
        category = ''
        for i in range(len(self.cat)):
            category += 'crime.category = {0} OR '.format(self.cat[i])
        category = category.strip('OR ')
        if len(category) > 1:
            category = '('+category+')'
            category += ' AND '
        query_string = base_query+event_join+crime_join+where+offprofile+vgroup+vsex+category
        # Check for lagging AND clause, just in case
        if 'AND ' in ''.join(query_string[-4:]):
            query_string = query_string.strip('AND ')
        # Fetch results
        results = cursor.execute(query_string)
        results = results.fetchall()
        conn.close()
        return results

    def load_offenders(self,results):
        valid_files = os.listdir(self.imgdir)
        # copy images matching our query to the Positive directory
        for pid in results:
            # try block because sometimes DLIB produces bad images
            try:
                filename = str(pid[0])+'.jpg'
                if filename in valid_files:
                    os.system('cp "'+self.imgdir+filename+'" "'+self.pdir+'"')
            except Exception as e:
                logging.info(e)
                continue

    def load_fitw(self,):
        
        # Copy normalized faces from the Faces in the Wild dataset
        # over to our Neutral folder
        source = os.path.join(self.fitw,self.dem)
        dest = os.path.join(self.ndir)
        files = os.listdir(source)
        for f in files:
            os.system('cp "'+os.path.join(source,f)+'" '+
                      os.path.join(dest,f))
        return None

    def load_ctl(self,):
        # Copy control mugshots of non-violent inmates
        conn = sqlite3.connect('inmates.db')
        cur = conn.cursor()
        query = cur.execute('SELECT id FROM inmates WHERE category="C"')
        results = query.fetchall() #Get 'civic' crimes only
        conn.close()
        results = [i[0]+'.jpg' for i in results] #add file extension
        allmugs = os.listdir(os.path.join(self.ctl,self.dem))
        for mug in allmugs:
            if mug in results:
                try:
                    os.system('cp "'+self.ctl+self.dem+'/'+mug+'" '+self.ndir)
                except:
                    logging.info(e)
        return None

    def load_goog(self,):
        # Copy faces found by searching for common names
        total = os.listdir(self.goog+self.dem)
        shuffle(total)
        sample_size = int(len(total)*self.sample)
        for x in total[:sample_size]:
            os.system('cp "'+self.goog+self.dem+'"/'+x+' '+self.ndir)
                    
    def load_race(self,):
        # Copy normalized faces found by google searches of specific
        # demographic groups
        os.system('cp "'+self.race+self.dem+'"/* '+self.ndir)

    def load_10k(self,):
        #Copy 10K faces
        os.system('cp "'+self.us10k+self.dem+'"/* '+self.ndir)
        
    def split_for_validation(self,):
        positive = os.listdir(self.pdir)
        neutral = os.listdir(self.ndir)
        shuffle(positive)
        shuffle(neutral)
        pindex = int(ceil(self.val*len(positive)))
        nindex = int(ceil(self.val*len(neutral)))
        for img in positive[:pindex]:
            os.rename(self.pdir+img,'Data/Offender/Validation/Positive/'+img)
        for img in neutral[:nindex]:
            os.rename(self.ndir+img,'Data/Offender/Validation/Neutral/'+img)
        
    
    def purge(self):
        # Deletes all images in the training folders.
        for root, dirs, files in os.walk('Data/Offender'):
            for file in files:
                os.remove(os.path.join(root,file))
        print('All image files removed from training directory')

class SOFTrainer:
    """
    Transfer learning model which extracts features from VGG Face and
    trains an MLP on top of the features. Data is sourced from the
    'Data/Offender' folder, which should be populated with images from the
    DataPrep class. The resulting model is saved and the training results
    are saved to a plot.

    This class is initialized with globally defined variables. The globals
    should be the only variables necessary to modify for normal usage.

    Currently, this is an adaptation of the transfer learning example from
    the Keras blog and thus some of the programming idioms may be
    incongruous. Future versions should eliminate waste and ambiguous name
    conventions.
    """

    def __init__(self):
        self.img_width,self.img_height = 224,224
        self.train_data = 'Data/Offender/Train'
        self.validation_data = 'Data/Offender/Validation'
        self.batch_size = 16
        self.datagen = ImageDataGenerator(rescale=1./255)
        self.num_train = sum([len(files) for r,d,files in os.walk(self.train_data)])
        self.num_valid = sum([len(files) for r,d,files in os.walk(self.validation_data)])
        self.top_model_weights_path = 'SOFC_weights_'+demographic_group+'.h5'
        self.model_path = 'SOFC_model_'+demographic_group+'.h5'
        self.epochs = 20
        
    def save_bottleneck_features(self):
        vggmodel = VGGFace(include_top=False,pooling='avg',input_shape=(224,224,3))
        if not os.path.isfile('SOFC_train_feats.npy'):
            train_generator = self.datagen.flow_from_directory(
                self.train_data,
                target_size=(self.img_width,self.img_height),
                batch_size=self.batch_size,
                class_mode=None,
                shuffle=False)
            
            nb_train_samples = len(train_generator.filenames)
            num_classes = len(train_generator.class_indices)
            predict_size_train = int(math.ceil(nb_train_samples/self.batch_size))

            train_features = vggmodel.predict_generator(train_generator,predict_size_train)
            np.save(open('SOFC_train_feats.npy','wb'),train_features)
            
        if not os.path.isfile('SOFC_validation_feats.npy'):
            validation_generator = self.datagen.flow_from_directory(
                self.validation_data,
                target_size=(self.img_width,self.img_height),
                batch_size=self.batch_size,
                class_mode=None,
                shuffle=False)

            nb_validation_samples = len(validation_generator.filenames)
            num_classes = len(validation_generator.class_indices)
            predict_size_validation = int(math.ceil(nb_validation_samples/self.batch_size))

            validation_features = vggmodel.predict_generator(validation_generator,
                                                            predict_size_validation)
            np.save(open('SOFC_validation_feats.npy','wb'),validation_features)

    def train_top(self):
        ### TRAIN DATA ###
        top_train_gen = self.datagen.flow_from_directory(
            self.train_data,
            target_size=(self.img_width,self.img_height),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False)
        num_train_samples = len(top_train_gen.filenames)
        # Cut off samples that aren't divisible into our batch size
        num_train_samples = num_train_samples - (num_train_samples % self.batch_size)
        num_classes = len(top_train_gen.class_indices)
        train_labels = top_train_gen.classes
        train_labels = train_labels[:num_train_samples]
        train_labels = to_categorical(train_labels,num_classes=num_classes)
        train_data = np.load(open('SOFC_train_feats.npy','rb'))

        ### VALIDATION DATA ###

        top_validation_gen = self.datagen.flow_from_directory(
            self.validation_data,
            target_size=(self.img_width,self.img_height),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False)
        num_validation_samples = len(top_validation_gen.filenames)
        # Same hack as above
        num_validation_samples = num_validation_samples - (num_validation_samples % self.batch_size)
        num_classes = len(top_validation_gen.class_indices)
        validation_labels = top_validation_gen.classes
        validation_labels = validation_labels[:num_validation_samples]
        validation_labels = to_categorical(validation_labels,num_classes=num_classes)
        validation_data = np.load(open('SOFC_validation_feats.npy','rb'))

        ### MODEL ###

        model = Sequential()
        model.add(Dense(512,input_shape=train_data.shape[1:],activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes,activation='softmax'))

        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        history = model.fit(train_data,train_labels,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_data=(validation_data,validation_labels))
        model.save_weights(self.top_model_weights_path)
        model.save(self.model_path)

        (eval_loss,eval_accuracy) = model.evaluate(validation_data,
                                                   validation_labels,
                                                   batch_size=self.batch_size,
                                                   verbose=1)

        print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
        print("[INFO] loss: {}".format(eval_loss))
        ####### PLOTTING ##########
        plt.subplot(211)  
        plt.plot(history.history['acc'])  
        plt.plot(history.history['val_acc'])  
        plt.title('model accuracy')  
        plt.ylabel('accuracy')  
        plt.xlabel('epoch')  
        plt.legend(['train', 'test'], loc='upper left')  

        # summarize history for loss  

        plt.subplot(212)  
        plt.plot(history.history['loss'])  
        plt.plot(history.history['val_loss'])  
        plt.title('model loss')  
        plt.ylabel('loss')  
        plt.xlabel('epoch')  
        plt.legend(['train', 'test'], loc='upper left')  
        plt.savefig(demographic_group+'-training plot') 


if __name__ == "__main__":
##    print('Initializing . . .')
##    D = DataPrep()
##    print('Purging old data')
##    D.purge()
##    results = D.query_offenders()
##    print('Loading positive samples')
##    D.load_offenders(results)
##    print('Loading 10K faces dataset')
##    D.load_10k()
####    print('Loading FITW dataset')
####    D.load_fitw()
####    print('Loading non-SOF mugshots')
####    D.load_ctl()
####    print('Loading faces from name searches')
####    D.load_goog()
####    print('Loading random faces')
####    D.load_race()
##    print('Creating validation dataset')
##    D.split_for_validation()
    print('Initializing trainer')
    S = SOFTrainer()
    print('Saving features . . .')
    S.save_bottleneck_features()
    print('Training top model . . .')
    S.train_top()
    print('Complete. Exiting . . .')
    quit()
