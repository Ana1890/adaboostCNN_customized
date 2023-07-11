
from constants import * 

import time

import numpy as np
from numpy.core.umath_tests import inner1d
from copy import deepcopy

##kerase & CNN:
#from keras import models as Models
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import LabelBinarizer
from constants import * 
# tf.config.run_functions_eagerly(True)
mean_loss = tf.keras.metrics.Mean(name='Loss_Mean')



earlystopper = EarlyStopping(patience=params['patience'], verbose=1),
checkpoint = ModelCheckpoint(filepath=params['name_modelo'] + f'{specific_time} modelo retinopatia.h5',
                             save_best_only=params['save_best_only'], verbose=1)

def get_recall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = TP / (Positives + K.epsilon())
    return recall


def get_precision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = TP / (Pred_Positives + K.epsilon())
    return precision


def get_f1_score(y_true, y_pred):
    precision, recall = get_precision(y_true, y_pred), get_recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

callbacks = [earlystopper, checkpoint]

def print_status_bar(iteration, total, loss, metrics=None, bar_size=45):
    '''This function prints out the loss + other metrics (if specified) one below another.
    iteration -> Epoch
    total     -> Total Epochs
    loss      -> Model loss for that epoch
    metrics   -> Calculated metrics'''
    
    metrics = ' - '.join(["{}: {:.4f}".format(m.name, m.result().numpy()) for m in [loss] + [metrics]])# + (metrics or [])
    end = "" if iteration < total else "\n"
    
    # \r ensures that the output is printed in the same line
    # print ("\r{}/{} - ".format(iteration, total) + metrics, end=end)
    
    # \r ensures that the output is printed in the same line
    # Change bar_size to get larger or smaller bars
    print ("\r[{}{}] - "
           .format(("=" * int(iteration/total*bar_size) + ">"), 
                   ("." * int((total-iteration)/total*bar_size))
           ) + metrics, end=end)
    

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)


      

class AdaBoostClassifier(object):
    '''
    Parameters
    -----------
    base_estimator: object
        The base model from which the boosted ensemble is built.

    n_estimators: integer, optional(default=50)
        The maximum number of estimators

    learning_rate: float, optional(default=1)

    algorithm: {'SAMME','SAMME.R'}, optional(default='SAMME.R')
        SAMME.R uses predicted probabilities to update wights, while SAMME uses class error rate

    random_state: int or None, optional(default=None)


    Attributes
    -------------
    estimators_: list of base estimators

    estimator_weights_: array of floats
        Weights for each base_estimator

    estimator_errors_: array of floats
        Classification error for each estimator in the boosted ensemble.

    Reference:
    1. [multi-adaboost](https://web.stanford.edu/~hastie/Papers/samme.pdf)

    2. [scikit-learn:weight_boosting](https://github.com/scikit-learn/
    scikit-learn/blob/51a765a/sklearn/ensemble/weight_boosting.py#L289)

    '''

    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state', 'epochs', 'sample_size']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        n_estimators = 10
        learning_rate = 1
        algorithm = 'SAMME.R'
        random_state = None
        #### CNN (5)
        epochs = 6

        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs: random_state = kwargs.pop('random_state')
            if 'sample_size' in kwargs: sample_size = kwargs.pop('sample_size')

            ### CNN:
            if 'epochs' in kwargs: epochs = kwargs.pop('epochs')


        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.algorithm_ = algorithm
        self.sample_size = sample_size
        self.random_state_ = random_state
        self.estimators_ = list()
        self.optimizer = tf.keras.optimizers.Adam(params['lr'])
        self.n_batch = 0
        
        self.train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        self.train_loss_metric = tf.keras.metrics.BinaryCrossentropy()
        self.val_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        self.val_loss_metric = tf.keras.metrics.BinaryCrossentropy()

        self.patience = 2
        self.wait = 0
        self.best = 0
        self.list_save = list()
  
        
        #self.config = config
        self.epochs= epochs

    def train_step(self, x, y, sample_weight):
        y = np.asarray(y).astype('float32').reshape((-1,1))
        y = tf.constant(y)
        with tf.GradientTape() as tape:
            logits = self.estimator(x, training=True)
            # print("label y pred: ", y, logits)
            loss_value = loss_fn(y, logits, sample_weight=sample_weight)
            # print('pérdida: ', loss_value)
        grads = tape.gradient(loss_value, self.estimator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.estimator.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        # print("métrica: ", self.train_acc_metric.result().numpy())
        self.train_loss_metric.update_state(y, logits)
        return loss_value

    def test_step(self, x, y):
        logits = self.estimator(x, training=False)
        self.val_acc_metric.update_state(y, logits)
        self.val_loss_metric.update_state(y, logits)
        

    def fit(self, training_generator, val_generator):
        self.batch_size = len(training_generator[0])
        
        for epoch in range(self.epochs):
            print("\nEntrenamiento de la epoch nro {0}".format(epoch))
            
            for i in range(len(training_generator)):
                # print("\nEntrenamiento del batch nro {0}".format(i))                
                sample_weights = np.ones(self.batch_size)
                
                
                sample_weights, loss_value = self.boost(training_generator, sample_weights)
                # print('Pérdida y métrica a printear: ', self.train_loss_metric.result().numpy(), self.train_acc_metric.result().numpy())
                print_status_bar(i * self.batch_size, len(training_generator)*self.batch_size, self.train_loss_metric, self.train_acc_metric)   

                self.n_batch += 1
                
            self.train_acc = self.train_acc_metric.result()
            self.train_loss = self.train_loss_metric.result()
            self.train_acc_metric.reset_states()
            self.train_loss_metric.reset_states()
    
                
            ## Validation
            for i, (X_batch_val, y_batch_val) in enumerate(val_generator):
                self.test_step(X_batch_val, y_batch_val)
                print_status_bar(i * self.batch_size, len(val_generator)*self.batch_size, self.val_loss_metric, self.val_acc_metric)   

                
            val_acc = self.val_acc_metric.result()
            val_loss = self.val_loss_metric.result()
            self.val_acc_metric.reset_states()
            self.val_loss_metric.reset_states()
            
            self.wait += 1
            if val_loss > self.best:
                self.best = val_loss
                self.wait = 0
                self.list_save = list()
            else:
                self.list_save.append(self.estimator)
                
            if self.wait >= self.patience:
                tf.keras.models.save_model(self.list_save[0], 'best_model.h5', save_format='h5')
                break
            
            
            
            self.n_batch = 0
            
        return self, self.estimator


    def boost(self, training_generator, sample_weights):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(training_generator, sample_weights)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(training_generator, sample_weights)
        
        
            
    def real_boost(self, training_generator, sample_weights):
        # estimator = deepcopy(self.base_estimator_)

        if len(self.estimators_) == 0:
            if self.n_batch==0:
                # Copy CNN to estimator:
                self.estimator = self.deepcopy_CNN(self.base_estimator_) # deepcopy of self.base_estimator_ # Agarra los pesos del modelo actual
        else: 
            if self.n_batch==0:
                self.estimator = self.deepcopy_CNN(self.estimators_[-1]) # deepcopy CNN # Agarra los pesos del anterior modelo
        if self.random_state_:
                self.estimator.set_params(random_state=1)
                
        X,y = training_generator[self.n_batch]

        # CNN (3) binery label:       
        lb=LabelBinarizer()
        y_b = lb.fit_transform(y)
        
        #print("Estimator nro: ", len(self.estimators_))

        if len(self.estimators_)>=1:
            y_pred = self.estimator.predict(X)
            y_pred_l = np.where(y_pred>0.5,1,0)
            incorrect = y_pred_l != y_b
            
            #print("old sample_weights: ", sample_weights)
            sample_weights = sample_weights*np.exp(-1. * self.learning_rate_ * (((2 - 1) / 2) *
                                                            inner1d(y_b, np.log(
                                                                y_pred)))) 

            sample_weight_sum = np.sum(sample_weights, axis=0)
            

            # if sample_weight_sum <= 0:
            #     return None, None, None

            # normalize sample weight
            sample_weights /= sample_weight_sum
        else: 
            sample_weights = np.ones(self.batch_size)
        
            

        #with tf.GradientTape() as tape:
        #print('New sample_weights: ', sample_weights)
        # hist = self.estimator.fit(X,y,sample_weight=sample_weights, epochs = 1)

        loss_value = self.train_step(X, y, sample_weights)

        
        
        
        # append the estimator
        if self.n_batch >= int(self.sample_size/self.batch_size)-1:
            print("Nuevo estimator..")
            self.estimators_.append(self.estimator) 
            if len(self.estimators_)>3:
                self.estimators_ = self.estimators_[-2:]
        

        return sample_weights, loss_value
    
    
    def deepcopy_CNN(self, base_estimator0):
        # Copy CNN (self.base_estimator_) to estimator:
        config=base_estimator0.get_config()
        estimator = Sequential.from_config(config) # Define el estimator

        weights = base_estimator0.get_weights()
        estimator.set_weights(weights) # Le atribuye al estimator los pesos de inicialización por defoult o los pesos del modelo anterior
        # estimator.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=[params['metrics'], get_recall, get_f1_score, get_precision])

        return estimator 

    




