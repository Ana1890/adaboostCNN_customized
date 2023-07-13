import datetime
from datetime import datetime


### CONSTANTS
fecha_actual = datetime.now()
str_date = fecha_actual.strftime('%Y.%m.%d')
specific_time = fecha_actual.strftime('%Y.%m.%d %H.%M')

### PATHS

# Kaggle
train_path = '/mnt/lxestudios/pacs/Eye/EyePacs/diabetic-retinopathy-detection/train/'
test_path = '/mnt/lxestudios/pacs/Eye/EyePacs/diabetic-retinopathy-detection/test/'

# labels
# root_path = 'C:/Users/UsuarioHI/Untitled Folder/'
root_path = '/mnt/lxestudios/pacs/Eye/EyePacs/diabetic-retinopathy-detection/'

# load model segmentation
root_model = '/mnt/lxestudios/pacs/Eye/Modelo Segmentacion/'

# Load segmented images
root_masks = '/mnt/lxestudios/pacs/Eye/pre-proc-eyepacs-propuesto/all_segmentacion/'

# save model 
models_path = '/mnt/lxestudios/pacs/Eye/Modelos/Modelos_retinopatia/Modelo_Binario/'

#### DICT
params = {'batch_size': 2,
          'n_channels': 3,
          'n_classes': 2,
          'shuffle_train': True,
          'shuffle_val': False,
          'test_size': 0.2,
          'stratify': True,
          'input_shape_model': (512,512,3),
          'dropout': 0.15,
          'output_activation':'sigmoid',
          'loss':'binary_crossentropy',
          'metrics': 'accuracy',
          'name_modelo':  models_path,
          'optimizer':'adam',
          'epochs':10,
          'patience':3,
          'save_best_only':True,
          'include_top':False,
          'weights':'imagenet',
          'model_name': 'VGG16',
          'trainable':True,
          'augment_train':True,
          'augment_val':False,
          'seed':42,
          'lr':0.0001,
          'masked': True,
          'proc': True,
         
          
         }

params_train = {'dim':params['input_shape_model'][:2],
               'shuffle':params['shuffle_train'],
               'batch_size': params['batch_size'],
               'n_channels': params['n_channels'],
               'n_classes': params['n_classes'],
                'augment': params['augment_train'],
                'masked': params['masked'],
                'proc': params['proc']
                
               }

params_valid = {'dim':params['input_shape_model'][:2],
               'shuffle':params['shuffle_val'],
               'batch_size': params['batch_size'],
               'n_channels': params['n_channels'],
               'n_classes': params['n_classes'],
                'augment': params['augment_val'],
                'masked': params['masked'],
                'proc': params['proc']
               }