import cv2
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

root_masks = '//lxestudios/pacs/Eye/pre-proc-eyepacs-propuesto/all_segmentacion/'

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
          'metrics': ['accuracy'],
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

def padding_image(img):

    if img.shape[0] > img.shape[1]: # Si el alto es mayor que el ancho
        faltante = img.shape[0] - img.shape[1]
        mitad = faltante//2
   
        img_border=cv2.copyMakeBorder( src=img, top=0, bottom= 0,left=mitad, right=mitad, borderType=cv2.BORDER_CONSTANT, 
                                   value=int(img.min()) )
        return img_border

    elif img.shape[1] > img.shape[0]: # Si el ancho es mayor que el alto
        faltante = img.shape[1] - img.shape[0]
        mitad = faltante//2
        
        img_border=cv2.copyMakeBorder( src=img, top=mitad, bottom=mitad, left=0, right=0, borderType=cv2.BORDER_CONSTANT, 
                                   value=int(img.min()) )

        return img_border
    
    elif img.shape[1] == img.shape[0]:
        return img
    
    
def recortar(img):
    
    #img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    row_center = img.shape[0]//2
    column_center = img.shape[1]//2
    
    # filas
    for i in range(row_center):
        
        if img[row_center + i,:,:].max() != 0:
            pass
        elif img[row_center + i,:,:].max() == 0:
            img = img[:row_center + i,:,:]
            break
            
    row_center = img.shape[0]//2
    column_center = img.shape[1]//2
        
    for i in range(row_center):
        if img[row_center - i,:,:].max() != 0:
            pass
        elif img[row_center - i,:,:].max() == 0:
            img = img[row_center - i:,:,:]
            break
            
    row_center = img.shape[0]//2
    column_center = img.shape[1]//2
    # columnas
    
    for i in range(column_center):
        if img[:,column_center + i,:].max() != 0:
            pass
        elif img[:,column_center + i,:].max() == 0:
            img = img[:, :column_center + i,:]
            break
            
    row_center = img.shape[0]//2
    column_center = img.shape[1]//2
        
    for i in range(column_center):
        if img[:,column_center - i,:].max() != 0:
            pass
        elif img[:,column_center - i,:].max() == 0:
            img = img[:,column_center - i:,:]
            break
        
    return img    

def relacionar_aspecto(img):
    """Recorte hacia adentro para mantener relación de aspecto """
    relacion = img.shape[0]/img.shape[1]
    
    if relacion < 1: # es más ancho que alto
        
        cantidad = img.shape[1]*0.5*(1-relacion)
        new_img = img[:,int(cantidad):(img.shape[1]-int(cantidad)),:]
        
    elif relacion > 1: # es más alto que ancho
        cantidad = img.shape[0]*0.5*(relacion-1)
        
        new_img = img[int(cantidad):(img.shape[0]-int(cantidad)),:]
    
    else: #no se recorta
        new_img = img.copy()
        
#     if new_img.shape[0]!=new_img.shape[1]:
#         print('aun no es cuadrada, shape 0: {} shape 1: {}'.format(new_img.shape[0], new_img.shape[1]))
        
    return new_img

def recorte_90p(mask, img):
    # Recorte al 90% de la máscara
    mask1_res = cv2.resize(mask,(460,460))

    r1 = 512/2
    r2 = 460/2

    rest = int(r1-r2)

    img_border_mask=cv2.copyMakeBorder(src=mask1_res, top=rest, bottom= rest,left=rest, right=rest, borderType=cv2.BORDER_CONSTANT, 
                                       value=int(mask1_res.min()) )

    masked = cv2.bitwise_and(img,img_border_mask)

    masked = masked/255
    
    return masked


def load_image(img_name, masked, proc):
    
    #print(img_name)
    img = cv2.imread(img_name) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = padding_image(img)
    
    if masked:
        img = cv2.resize(img,(768,768))
        # Enmascarar y segmentar
        res = [idx for idx, val in enumerate(img_name) if val in img_name[:idx] and '/' in val]
        mask_name = img_name[res[-1] + 1:]
        mask = cv2.imread(root_masks + mask_name)
        sup = cv2.bitwise_and(np.array(img),mask)
        img = recortar(sup)
        
        
        if proc:
            # Añadir el filtro
            img = padding_image(img)
            img = cv2.resize(img,params['input_shape_model'][:2])
            img = cv2.addWeighted( np.array(img),4, cv2.GaussianBlur( np.array(img) , (0,0) , 10) ,-4 ,110)
            
            # Aplico un filtro erode porque tal vez al recortarlo nuevamente no vuelve a su dibujo original
            kernel = np.zeros((7,7),np.uint8) # creamos el elemento estructurante 
            kernel[2:5,:]=1
            kernel[:,2:5]=1
            mask_er = cv2.erode(mask,kernel,iterations = 1)
    
            mask_rec = recortar(mask_er)    
            mask_rec = padding_image(mask_rec)
            mask_res = cv2.resize(mask_rec,params['input_shape_model'][:2])
            
            img = recorte_90p(mask_er,img)
            
            return img
        

        # volver a la imagen cuadrada si aún no lo es para mantener relación de aspecto, por lo general, se achica
        #img = relacionar_aspecto(img)
        img = padding_image(img)
        img = cv2.resize(img,params['input_shape_model'][:2])
        img = img/255
    else: 
        img = cv2.resize(img,params['input_shape_model'][:2])
        img = img/255
        
    return img