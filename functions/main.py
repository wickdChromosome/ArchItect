#!/usr/bin/env python3 

from multiprocessing.pool import ThreadPool
from keras.models import Input, Model
from keras.layers import Dropout
import numpy as np
import random
from keras.layers import concatenate
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
import threading
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from subprocess import check_output
from keras.models import model_from_json
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
import logging
import multiprocessing


optimFuncList = ['sgd', 'rmsprop','adagrad','adadelta','adam','adamax','nadam','tfoptimizer']
#lossFuncList = ['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','mean_squared_logarithmic_error','squared_hinge', 'hinge','categorical_hinge','logcosh','categorical_crossentropy','sparse_categorical_crossentropy','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity']
activFuncList = ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
lossFuncList = ['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','mean_squared_logarithmic_error','squared_hinge', 'hinge','categorical_hinge','logcosh','sparse_categorical_crossentropy','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity']

#define logger
logging.basicConfig(level=logging.DEBUG,filename='../logs/debug.log' ,format='%(relativeCreated)6d %(threadName)s %(message)s')


#define training env to evolve in
class environment():

    #number of errors until a neural net is overfitted (when solving for optimal layer number)
    errorLim = 4
    #start out with using 10% of data
    startingSampleSize = 10
    currentPlayers = [] #all current nets surviving
    #define number of children each round
    childNum = 5
    currentPlayerLimit = 0.5 #concurrent neural net limit (by RMSE, top % survive, other die)
    #keep high score
    currentHighScore = math.inf #lowest RMSE value in env
    minScoreDelta = 0.01
    #initital ancestors spawned
    initSpawnNum = 10
    #define path for data
    dataPath = '../data/btc_2016-2018-3-1.csv'
    #define number of errors for layer opt
    errorLim = 4
    dataset = [] #dataset to be later imported
    scaler = [] #used when preprocessing csv
    currentInstances = [] #used to store processes for each net
    IOTries = 5 #number of tries b4 killing thread for IO failure

    #used to give an ID to all neural nets
    globalID = 0

    #used to store last high score for comparison
    prevHighScore = math.inf

    #use a commonly used neural net to calculate how much data to use for
    #sampling
    def getSampleSize(self):

        #gradient descent
        ready = False
        prevRMSE = math.inf
        isFirstIter = True
        direction = 10 #start out by increasing by 10%
        deltaRMSE = math.inf
        counter = 0
        testNet = environment.ancestor(True)
        startingSampleSize = environment.startingSampleSize

        logging.debug('Starting to optimize for starting sample size')

        while not ready:

            logging.debug('Current startingSampleSize: ' + str(startingSampleSize))

            #if we have at least 1 iteration
            if counter >= 1:

                startingSampleSize = environment.startingSampleSize
                startingSampleSize += direction #modify sample size by direction
                

                #define commonly used neural net
                testNet.train()

                #if we are getting better, dont change direction
                if testNet.bestRMSE <= prevRMSE:

                    #measure change in RMSE
                    deltaRMSE = (testNet.bestRMSE-prevRMSE)/prevRMSE

                    logging.debug('Calculated change in RMSE: ' + str(deltaRMSE))

                    #if change in data% is less than 0.5%, we are also done
                    #if change in RMSE less than 1%, we are done
                    if deltaRMSE < testNet.bestRMSE/100 or abs(direction) < 0.5 :
                        
                        ready = True

                #if our RMSE is increasing
                else:
                
                    #reverse direction, divide movement size by 2
                    direction = -direction/2
 

            #if its the first iteration
            else:

                logging.debug('Entering first loop')    
                counter += 1

                #iterate sampleSize, see what happens
                environment.startingSampleSize += direction

                #set the first, initial random layer
                testNet.train()

                prevRMSE = testNet.bestRMSE
               
                logging.debug('Calculated starting RMSE: '+ str(prevRMSE))
   
    #ancestors are basic neural nets
    #all us the startingSamplesize for evaluating best setup
    class ancestor():

        class layer(): 

            #randomizes layer props as a start
            def __init__(self, isCalibration=False):

                if isCalibration:

                    #number of hidden layers in deep net, between 1 and 64
                    self.layerNum = random.randint(1,65)
                    self.activFunc = 'softmax'               

                else:

                    self.activFunc = random.choice(activFuncList)
                    #number of hidden layers in deep net, between 1 and 64
                    logging.debug('Mutated a new random layer with ' + str(self.activFunc))      

        #test a dataset based on setup
        def test(self):
          
            scaler=environment.scaler #make sure that data stays untouched 
            colNum = 4
            np.random.seed(7)
            dataset=environment.dataset
            # split into train and test sets
            train_size = int(len(dataset) * 0.8)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
            

            # reshape into X=t and Y=t+1
            look_back = 1
            trainX, trainY = environment.create_dataset(train, look_back)
            testX, testY = environment.create_dataset(test, look_back)


            # reshape input to be  [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
            print(trainX)

            # create and fit the LSTM network

            model = Sequential()
            #input columns
            model.add(LSTM(colNum, input_dim=look_back))
            
            
            #iterate through hidden layers
            for layer in self.layers:
               
                #vary activation function and number of hidden layers for each net stack
                model.add(Dense(1, activation=layer.activFunc)) #dense layers to be varied
           
            model.compile(loss=self.lossFunc, optimizer=self.optimFunc)     
            history= model.fit(trainX, trainY,validation_split=0.33, nb_epoch=100, batch_size=10000)


            # Plot training
            """
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('epoch')
            plt.xlabel('rmse')
            plt.legend(['loss', 'val_loss'], loc='upper right')
            plt.show()
            """
            # make predictions
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            # Get something which has as many features as dataset
            trainPredict_extended = np.zeros((len(trainPredict),3))
            # Put the predictions there
            trainPredict_extended[:,2] = trainPredict[:,0]
            # Inverse transform it and select the 3rd column.
            trainPredict = scaler.inverse_transform(trainPredict_extended) [:,2]
            print(trainPredict)
            # Get something which has as many features as dataset
            testPredict_extended = np.zeros((len(testPredict),3))
            # Put the predictions there
            testPredict_extended[:,2] = testPredict[:,0]
            # Inverse transform it and select the 3rd column.
            testPredict = scaler.inverse_transform(testPredict_extended)[:,2]


            trainY_extended = np.zeros((len(trainY),3))
            trainY_extended[:,2]=trainY
            trainY=scaler.inverse_transform(trainY_extended)[:,2]


            testY_extended = np.zeros((len(testY),3))
            testY_extended[:,2]=testY
            testY=scaler.inverse_transform(testY_extended)[:,2]


            # calculate root mean squared error
            trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY, testPredict))
            print('Test Score: %.2f RMSE' % (testScore))

            # shift train predictions for plotting
            trainPredictPlot = np.empty_like(dataset)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back, 2] = trainPredict

            # shift test predictions for plotting
            testPredictPlot = np.empty_like(dataset)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, 2] = testPredict

            tries = 0
            done = False 
            while not done:

                #try catch due to several threads
                try:

                    model_json = model.to_json()
                    with open("../data/model" + str(self.ID) + ".json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    model.save_weights("../data/model" + str(self.ID) + ".h5")
                    logging.debug("Saved model to disk")

                    json_file = open('../data/model' + str(self.ID) + '.json', 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    loaded_model = model_from_json(loaded_model_json)
                    # load weights into new model
                    loaded_model.load_weights("../data/model" + str(self.ID) + ".h5")
                    
                    done = True

                except:

                    triesLeft = environment.IOTries - tries
                    
                    if triesLeft == 0:

                        logging.debug('Ran out of tries, terminating thread..')
                        

                    else:

                        logging.debug('Error with IO, trying again ' + str(triesLeft) + ' more times')

            return testScore 
            #plot
           
        '''
            series=plt.plot(scaler.inverse_transform(dataset)[:,2])
            prediccion_entrenamiento,=plt.plot(trainPredictPlot[:,2],linestyle='--')
            prediccion_test,=plt.plot(testPredictPlot[:,2],linestyle='--')
            #,'actual','guess'
            plt.legend([series],['series','trained','tested'])
            plt.yscale('log')
            plt.show()
        '''

        #test a config, return 
        #sampleSize is a % value used for evaluating children
        #layerProps is {layerNum:[activFunc, lossFunc, optimFunc], etc}
        def train(self):

            #istoodeep gets set to true if errorcount < errorLim
            isTooDeep = False
            errorcount = 0
            prevRMSE = math.inf
            bestSetup = [] #placeholder var for best self obj

            logging.debug('Optimizing layer number for net..')

            #iterate depth until rmse starts decreasing 
            while not isTooDeep:

                #if we are calibrating training data size
                if self.isCalibrator:

                    logging.debug('Setting up net for calibrating starting samples..')

                    #reproducable layer setup
                    if len(self.layers) == 0:
                        newLayer = self.layer(True)
                        self.layers.append(newLayer)

                    #reproducable neural net setup
                    self.lossFunc = 'mean_squared_error'         
                    self.optimFunc = 'sgd'

                    logging.debug('Set up new net with ' + self.optimFunc + ' ' + self.lossFunc)
                    logging.debug('test net has # of layers:' + str(len(self.layers)))

                else:

                    logging.debug('Spawning live net to evolve layers..') 

                    #spawn new random layer
                    newLayer = self.layer()
                    self.layers.append(newLayer) 
 
                    logging.debug('Set up new net with ' + self.optimFunc + ' ' + self.lossFunc)
                    logging.debug('New net has # of layers:' + str(len(self.layers)))


                #get new RMSE
                rmse = self.test()
                prevRMSE = rmse
                logging.debug("Found new RMSE: " + str(prevRMSE))

                #check if rmse > prev_rmse
                if rmse >= prevRMSE:
                    
                    #first time error happens, save it as basic best rmse
                    if errorcount == 0:
                    
                        self.bestRMSE = rmse
                        logging.debug("Saved first error's RMSE(" + str(rmse) + ") as " + str(self.bestRMSE))

                    logging.error('No optimization w/ additional layer')
                    errorcount += 1

                    if errorcount > environment.errorLim:

                        logging.error('Reached error limit')
                        isTooDeep = True

                if rmse < prevRMSE:

                    #save best version
                    with open('../data/mostoptimalNet' + str(self.ID)  + '.pkl', 'wb') as output:
                        
                        pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
                        self.bestRMSE = rmse

                prevRMSE = rmse

            try:


                #load most optimal config
                with open('../data/mostoptimalNet' + str(self.ID)  + '.pkl', 'rb') as input:

                    self = pickle.load(input)

            except:


                import pdb; pdb.set_trace()
                #if IOerror
                logging.debug("IOerror occurred when loading most optimal net version")


        def spawn(self):

            #define starting layer
            startingLayer = self.layer()

            #add first layer to layer cont
            self.layers = [startingLayer]
           
            #now get best RMSE for network
            self.train() 
 
    
        #randomly creates an ancestor and randomizes it
        def __init__(self, isCalibrator=False):

            if isCalibrator:
            
                self.isCalibrator = True
            
            else:

                self.isCalibrator = False
                logging.debug('Created non-calibrator ancestor')


            self.layers = []
            self.bestRMSE = math.inf 
            self.lossFunc = ''
            self.optimFunc = ''

            #randomize net's properties
            self.lossFunc = random.choice(lossFuncList)
            self.optimFunc = random.choice(optimFuncList) 

            environment.globalID += 1
            self.ID = environment.globalID

        #randomizes some property of one the layers,
        #or the net itself
        def randomizeProp(self):

            choices = [layerProfList, LossFuncList, activFuncList, self.layerNum]
            #randomize a single property for a random layer
            randomizedProp = random.choice(choices)

            if randomizedProp in activFuncList:

                 self.activFunc = randomizedProp

            elif randomizedProp in lossFuncList:

                 self.network.lossFunc = randomizedProp

            elif randomizedProp == self.layerNum:

                self.layerNum = random.randint(1,65)

            else:

                 self.network.optimFunc = randomizedProp






    #children are spawned
    class child(ancestor):

        def __init__(self, ancestor):

            #make logger for child

            #set RMSE to initial value
            self.bestRMSE = math.inf

            self.parent = ancestor

            #inherit layers of parent
            self.layers = self.parent.layers        

        def spawn(self):
   
            #randomize some aspect of parent for a single layer or several layers
            #random number of layers to randomize
            layerNumRandomize = random.randint(1, len(self.layers) + 1)
            #choose layers to randomize, then DO IT
            for layer in range(layerNumRandomize):

                #choose a layer to randomize
                layerChoice = random.choice(self.layers)
                layerChoice.randomizeProp() #now randomize some property of that layer


            #now get bestRMSE for net
            self.train()


    #dataPath is a string to csv
    def importData(self):

        # convert an array of values into a dataset matrix
        # fix random seed for reproducibility

        environment.dataframe = pd.read_csv( self.dataPath, engine='python')
        environment.dataframe = environment.dataframe[["ema14", "ema48", "price"]]
        print(environment.dataframe)
        environment.dataset = environment.dataframe.values
        print(environment.dataset)
     
        environment.scaler = MinMaxScaler(feature_range=(0,1))
        environment.dataset = environment.scaler.fit_transform(environment.dataset)

    def create_dataset(dataset, look_back=1):
       
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 2])
        return np.array(dataX), np.array(dataY)


    #produce children, if chosen
    def reproduce(player):

        #number of children defined by environment
        for childNumber in range(numChildren):
           
            offSpring = child(player)
   
    #if no longer good enough to survive
    def die(self, player):

       self.currentPlayers.remove(player) 




    #imports data, then lets evolution begin
    def __init__(self):

        

        #first import the data
        self.importData()

        #now get the sample size used for evaluation (modifies startingSampleSize)
        #self.getSampleSize()

        logging.debug('Imported and formatted training data')
        logging.debug('Found sampling size to be ' + str(self.startingSampleSize))

        #create ancestors
        for ancestorNum in range(environment.initSpawnNum):

            newAncestor = self.ancestor()#the actual descriptor for the process
            self.currentPlayers.append(newAncestor)
            logging.debug("Initialized ancestor # " + str(ancestorNum))
           
        logging.debug('About to start Ancestor Processes..')

        #now start them
        for ancestorPlayer in self.currentPlayers:
 
            ancestorProcess = multiprocessing.Process(
                    target=ancestorPlayer.spawn,
                    args=[]               
                    )    
            ancestorProcess.start()
            logging.debug('Here is the ancestor: ' + str(ancestorProcess))
    
            #add ancestor to environment
            self.currentInstances.append(ancestorProcess)

        #finish up
        for ancestor in self.currentInstances:

            ancestor.join()
            logging.debug("Done with Ancestor: " + str(ancestor))    

        #see current players in this round
        logging.debug("Current players in this round: " + str(environment.currentPlayers))

        logging.debug("Entered optimization loop")

        #lets keep training until minima reached
        done = False
        while not done:


            logging.debug("Entered optimization loop")

            #get array of RMSE values for current players
            rmseList = []
            for player in self.currentPlayers:
                
                #get top RMSE 
                rmseList.append(player.bestRMSE)
                logging.debug("Added new rmse score: " + str(player.bestRMSE))

            #now evaluate current players, and put new currentHighScore if found        
            numPlayers = int(self.currentPlayerLimit * len(rmseList)) #players in next round
            #iterate until we fill up the max number of players
            
            
            
            #import pdb; pdb.set_trace()
            topPlayers = []
            for topPlayerNum in range(numPlayers):

                #keep getting top players 
                topPlayer = min(rmseList)
                #remove top player from prev list
                rmseList.remove(topPlayer)
                topPlayers.append(topPlayer)

            #remove rest
            for player in self.currentPlayers:

                #if player is not one of the best, die
                if player not in topPlayers:

                    self.die(player)
                    logging.debug(str(player) + " is no more")


                #if its a top player, let it have children
                else:

                    self.reproduce(player)
                    logging.debug(str(player) + " just had children")

    
            #if player's value smaller than min rmse value in list, it survives
            minCost = min(rmseList)
            logging.debug("The top score is: " + str(minCost))
    
            #now take list of best players and spawn a new instance of them
            for player in self.currentPlayers:

                #now ancestors are treated as children as well 
                playerInstance = multiprocessing.Process(
                        target=player.spawn,
                        args=[]               
                        )    
                playerInstance.start()
                logging.debug(str(player) + " started")
                  
            for playerInstance in self.currentInstances:

                playerInstance.join()
            
            #check if the top RMSE changes enough
            #change in RMSE




            #if its first run
            if self.prevHighScore == math.inf:

    
                self.prevHighScore = self.currentHighScore
                logging.debug("Set prevHighScore to init value")


            delRMSE =(self.prevHighScore - self.currentHighScore)/self.currentHighScore 
            if delRMSE < self.minScoreDelta: #less than some change in RMSE
            
                done = True                
                logging.debug("The best player has been found")

    


env1 = environment()
