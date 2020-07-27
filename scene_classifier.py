import os
import json
import numpy as np
import typing
import tensorflow as tf
import efficientnet.tfkeras as effn

AUTO = tf.data.experimental.AUTOTUNE


def prepare_pred_dataset_from_fps(list_of_fp : [list], BATCH_SIZE : int = 32):
    
    """
    [ Reads a list of image filepaths and preprocess them for the image classifer model.
      Order of the list] of filepaths is preserved.]

    Args:
        list_of_fp ([list]): [list of image filepaths]
        BATCH_SIZE (int): BATCH_SIZE for prediction.
        Lower the batch size if run into OOM during prediction

    Returns:
        [Tensorflow Dataset]
    """    
 
    def read_decode_img(img_fp : str):
        # read, decode and preprocess single image filepath
        img = tf.io.read_file(img_fp)
        img = tf.io.decode_jpeg(img,try_recover_truncated=True)
        img = tf.image.resize(img,[260,260])/255 # normalise pixel values
        return img
    
    img_fp_ds = tf.data.Dataset.from_tensor_slices(list_of_fp)
    img_ds = img_fp_ds.map(read_decode_img,num_parallel_calls=AUTO) #map processing function to all images
    img_ds = img_ds.batch(BATCH_SIZE).prefetch(AUTO)
    return img_ds

def decode_prediction(pred_array,top_n, thresohold):
    
    """[Decode n x 22 predicton array from the classifier model]

    Args:
        pred_array ([np.array]): [raw prediction array from model prediction]
        
        top_n (int, optional): 
        [top_n prediction of the multilabel branch, regardless the confidence level of individual prediction]. 
        Defaults to 5.
        
        thresohold (float, optional): [Thresohold of multilabel prediction]. Defaults to 0.5.
    Returns:
    [three list of decoded predictions]: 
    [Decoded predictions would return THREE lists in form of:
    [threeclass_pred_n_prob_paired],[multilabel_pred_top_n], [multilabel_pred_over_threshold]
    """    
    
    def multilabel_over_threshold_predictions(multilabel_pred_array,thresohold):
        # map labels to predictions > thresohold
        idx_with_conf = np.where(multilabel_pred_array>thresohold)[0] # find the index of element > thresohold
        pred_label = list(map(multilabel_class_index.get, idx_with_conf)) 
        label_prob = multilabel_pred_array[idx_with_conf]
        return dict(sorted(list(zip(pred_label,np.round(label_prob,4))),key = lambda x: x[1])[::-1])  # sort by the prob of (label,prob) pair and reverse it
    
    def multilabel_top_n_predictions(multilabel_pred_array, top_n):
        # map labels to top 5 prediction
        sorted_top5_pred = multilabel_pred_array.argsort()[::-1][:top_n]
        pred_label = list(map(multilabel_class_index.get, sorted_top5_pred))
        label_prob = multilabel_pred_array[sorted_top5_pred]
        return dict(list(zip(pred_label,np.round(label_prob,4))))

    threeclass_class_index = {0:'nature',1:'indoor',2:'ambiguous'} 
    multilabel_class_index = {0: 'indoor',
                             1: 'outdoor, natural',
                             2: 'outdoor, man-made',
                             3: 'shopping and dining',
                             4: 'workplace (office building, factory, lab, etc.)',
                             5: 'home or hotel',
                             6: 'transportation (vehicle interiors, stations, etc.)',
                             7: 'sports and leisure',
                             8: 'cultural (art, education, religion, millitary, law, politics, etc.)',
                             9: 'water, ice, snow',
                             10: 'mountains, hills, desert, sky',
                             11: 'forest, field, jungle',
                             12: 'man-made elements',
                             13: 'transportation (roads, parking, bridges, boats, airports, etc.)',
                             14: 'cultural or historical building/place (millitary, religious)',
                             15: 'sports fields, parks, leisure spaces',
                             16: 'industrial and construction',
                             17: 'houses, cabins, gardens, and farms',
                             18: 'commercial buildings, shops, markets, cities, and towns'}
    
    # decode three class classifier
    threeclass_pred = np.array([p[:3]for p in pred_array]) # first 3 element = predicion of the 3 class classifier
    threeclass_pred_argmax = threeclass_pred.argmax(axis=-1) # take the highest probability of 3 class as predicted class
    threeclass_pred_label = list(map(threeclass_class_index.get,threeclass_pred_argmax)) # map class index to label
    threeclass_pred_prob = [threeclass_pred[i][threeclass_pred_argmax[i]] for i in range(len(threeclass_pred_argmax))] # retrieve the probability of the 
    threeclass_pred_n_prob_paired = list(zip(threeclass_pred_label,threeclass_pred_prob)) #a pair of predicted label and its corresponding probability 
    
    # decode multilabel class classifier
    multilabel_pred = np.array([p[3:]for p in pred_array]) # forth to nineteenth element as 19 classes multilabel
    multilabel_pred_over_threshold = [multilabel_over_threshold_predictions(p,thresohold) for p in multilabel_pred] # multilabel predictions over threshold (0.5)
    multilabel_pred_top_n = [multilabel_top_n_predictions(p,top_n) for p in multilabel_pred] # top n multilabel predictions 
    
    return threeclass_pred_n_prob_paired,multilabel_pred_top_n, multilabel_pred_over_threshold

def run_classifier(image_fps, model_path, top_n = 5, thresohold = 0.5):
    pred_ds = prepare_pred_dataset_from_fps(image_fps)
    scenery_model = tf.keras.models.load_model(model_path)
    pred_vector = scenery_model.predict(pred_ds,verbose=1) # the prediction vector is a n x 22 array, n as in number of images predicted, 22 as 3class + 19 classes multilabel
    pred1,pred2,pred3 = decode_prediction(pred_vector,top_n, thresohold)
    return pred1,pred2,pred3  # [threeclass_pred_n_prob_paired],[multilabel_pred_top_n],[multilabel_pred_over_threshold]