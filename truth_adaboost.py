import numpy as np
from numpy.core.umath_tests import inner1d
from copy import deepcopy

##kerase & CNN:
#from keras import models as Models
from keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import LabelBinarizer


from constants import * 

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
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state', 'epochs']
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
            ### CNN:
            if 'epochs' in kwargs: epochs = kwargs.pop('epochs')


        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.algorithm_ = algorithm
        self.random_state_ = random_state
        self.estimators_ = list()
        self.estimator_weights_ = np.zeros(self.n_estimators_)
        self.estimator_metrics_ = np.ones(self.n_estimators_)
        
        #self.config = config
        self.epochs= epochs


    def fit(self, training_generator, valid_generator, class_weights):
        
        for iboost in range(self.n_estimators_):
            print("Entrenamiento del estimator nro {0}".format(iboost))
            if iboost == 0:
                class_weights = class_weights

            class_weights, estimator_weight, estimator_metric = self.boost(training_generator, valid_generator, class_weights)

            # append error and weight
            self.estimator_metrics_[iboost] = estimator_metric
            self.estimator_weights_[iboost] = estimator_weight

        return self


    def boost(self, training_generator, valid_generator, class_weights):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(training_generator, valid_generator, class_weights)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(training_generator, valid_generator, class_weights)
    
            
    def real_boost(self, training_generator, valid_generator, class_weights):
        # estimator = deepcopy(self.base_estimator_)

        if len(self.estimators_) == 0:
            # Copy CNN to estimator:
            estimator = self.deepcopy_CNN(self.base_estimator_) # deepcopy of self.base_estimator_ # Agarra los pesos del modelo actual
        else: 
            estimator = self.deepcopy_CNN(self.estimators_[-1]) # deepcopy CNN # Agarra los pesos del anterior modelo
        if self.random_state_:
                estimator.set_params(random_state=1)
        
        # # CNN (3) binery label:       
        # lb=LabelBinarizer()
        # y_b = lb.fit_transform(y)
        
        #with tf.GradientTape() as tape:
        hist = estimator.fit_generator(generator= training_generator, epochs = self.epochs, 
                             validation_data=valid_generator, callbacks=callbacks)#, class_weight=class_weights)
        
        min_val_loss = hist.history['val_loss']
        min_val_loss = min(min_val_loss)
        
        f1_score = hist.history['get_f1_score'][int(min_val_loss)]
        estimator_metric = f1_score
        
        # y_pred = estimator.predict(X)
        # y_pred_l = np.where(y_pred>0.5,1,0)
        # incorrect = y_pred_l != y_b
        
        # print("y_pred: \n", y_pred, "y_true: \n", y_b)
        # print("incorrect: \n", incorrect, " Sample weight: \n", sample_weight)
        
        #estimator_error = np.dot(incorrect.reshape(1,-1), sample_weight) / np.sum(sample_weight, axis=0)

        # print("estimator_error: ", estimator_error)
        # # if worse than random guess, stop boosting
        # if estimator_error >= 1.0 - 1 / self.n_classes_:
        #     return None, None, None

        # sample_weight = sample_weight*np.exp(-1. * self.learning_rate_ * (((2 - 1) / 2) *
        #                                                 inner1d(y_b, np.log(
        #                                                     y_pred)))) 
        for k,v in class_weights.items():
            class_weights[k] = v*np.exp(-1. * self.learning_rate_ * (((2 - 1) / 2) *
                                                        f1_score))

        class_weights_sum = np.sum(list(class_weights.values()),axis=0)
        

        if class_weights_sum <= 0:
            return None, None, None

        # normalize sample weight
        for k,v in class_weights.items():
            class_weights[k] = v/class_weights_sum

        # append the estimator
        self.estimators_.append(estimator) 

        return class_weights, 1, estimator_metric
    
    
    def deepcopy_CNN(self, base_estimator0):
        # Copy CNN (self.base_estimator_) to estimator:
        config=base_estimator0.get_config()
        estimator = Sequential.from_config(config) # Define el estimator

        weights = base_estimator0.get_weights()
        estimator.set_weights(weights) # Le atribuye al estimator los pesos de inicialización por defoult o los pesos del modelo anterior
        estimator.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=[params['metrics'], get_recall, get_f1_score, get_precision])

        return estimator 

    def discrete_boost(self, X, y, sample_weight):
    # estimator = deepcopy(self.base_estimator_)
           
        if len(self.estimators_) == 0:
            #Copy CNN to estimator:
            estimator = self.deepcopy_CNN(self.base_estimator_) # deepcopy of self.base_estimator_
        else: 
            estimator = self.deepcopy_CNN(self.estimators_[-1]) # deepcopy CNN
        
        if self.random_state_:
            estimator.set_params(random_state=1)

#        estimator.fit(X, y, sample_weight=sample_weight)
        # CNN (3) binery label:       
        lb=LabelBinarizer()
        y_b = lb.fit_transform(y)
        estimator.fit(X, y_b, sample_weight=sample_weight, epochs = self.epochs, batch_size = self.batch_size)
        y_pred = estimator.predict(X)
        
        #incorrect = y_pred != y
        # (4) CNN :
        y_pred_l = np.argmax(y_pred, axis=1)
        incorrect = y_pred_l != y
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        # update estimator_weight
        estimator_weight = self.learning_rate_ * (np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

        if estimator_weight <= 0:
            return None, None, None

        # update sample weight
        sample_weight *= np.exp(estimator_weight * incorrect)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, estimator_weight, estimator_error


