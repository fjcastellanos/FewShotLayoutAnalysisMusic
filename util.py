import utilIO
import random
import numpy as np
import utilConst
import tensorflow as tf
import cv2
import math


def create_Validation_and_Training_partitions(list_src_train, list_gt_train, pages_train=None):
    
    corpora = utilIO.match_SRC_GT_Images(list_src_train, list_gt_train)
    random.seed(78)
    random.shuffle(corpora)
    num_val_images = math.ceil(0.2*len(corpora))

    if len(list_src_train) == 1:
        val_data = corpora[0:1]    
        train_data = corpora[0:1]    
        return train_data, val_data

    val_data = corpora[0:num_val_images]

    if pages_train is None or pages_train == -1:
        pages_train = len(corpora)-num_val_images
    else:
        val_data = val_data[0:min(pages_train, len(val_data))]
    assert(pages_train <= (len(corpora)-num_val_images))
    train_data = corpora[num_val_images:num_val_images+pages_train]

    return train_data, val_data


def getInputShape(config):
    return (config.win_w, config.win_w, utilConst.kNUMBER_CHANNELS)


def appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks):
    gr_sample = gr[
            row : row + patch_height, col : col + patch_width
        ]  # Greyscale image
    gt_sample = gt[
        row : row + patch_height, col : col + patch_width
    ]  # Ground truth
    gr_chunks.append(gr_sample)
    gt_chunks.append(gt_sample)



def extractSequentialSamplesClass(gr, gt, window_w, window_h, batch_size, idx_starting_patch, gr_chunks, gt_chunks, regions_mask, augmentation_types):
    ROWS = gt.shape[0]
    COLS = gt.shape[1]
    
    min_rate_annotated_pixels = 0.0025

    patch_counter = 0

    #print("extractSequentialSamplesClass")
    #print("Starting: " + str(idx_starting_patch))
    
    for row in range(window_w//2, ROWS+window_w//2-1, window_w):
        for col in range(window_h//2, COLS+window_h//2-1, window_h):
            row = min(row, ROWS-window_w//2)
            col = min(col, COLS-window_h//2)
            
            gr_sample = gr[
                row-window_w//2 : row-window_w//2 + window_w, col-window_h//2 : col-window_h//2 + window_h
            ]
            gt_sample = gt[
                row-window_w//2 : row-window_w//2 + window_w, col-window_h//2 : col-window_h//2 + window_h
            ]
            regions_mask_sample = regions_mask[
                row-window_w//2 : row-window_w//2 + window_w, col-window_h//2 : col-window_h//2 + window_h
            ]
            
            if (np.sum(gt_sample == 1) > batch_size):
                current_rate_annotated_pixels = np.sum(gt_sample == 1) / (window_h*window_w)
                            
                if current_rate_annotated_pixels >= min_rate_annotated_pixels:
                    if patch_counter >= idx_starting_patch:
                        #print (patch_counter)
                        #print (str(row) + "-" + str(col))
                    
                        gr_aug_sample, gt_aug_sample, regions_mask_aug_sample, applied_augmentations = apply_random_augmentations(gr_sample, gt_sample, regions_mask_sample, augmentation_types, window_w, window_h)
                        gr_chunks.append(gr_aug_sample)
                        gt_chunks.append(gt_aug_sample)
                    
                    patch_counter += 1

                    if patch_counter >=(idx_starting_patch + batch_size):
                        return patch_counter
    return patch_counter
        

def extractRandomSamplesClass(gr, gt, patch_width, patch_height, batch_size, gr_chunks, gt_chunks, regions_mask, augmentation_types):

    min_rate_annotated_pixels = 0.0025
    potential_training_examples = np.where(gt == 1)

    num_coords = len(potential_training_examples[0])
    
    tries = 0
    MAX_TRIES = 100

    if num_coords >= batch_size:
        num_samples = 0
        while (num_samples < batch_size):
            idx_coord = random.randint(0, num_coords-1)
            row = potential_training_examples[0][idx_coord]
            col = potential_training_examples[1][idx_coord]

            row = max(patch_width//2+1, row-100)
            col = max(patch_height//2+1, col-50)

            row = min(gr.shape[0]-patch_width//2-1, row)
            col = min(gr.shape[1]-patch_height//2-1, col)

            gr_sample = gr[
                row-patch_width//2 : row-patch_width//2 + patch_width, col-patch_height//2 : col-patch_height//2 + patch_height
            ]
            gt_sample = gt[
                row-patch_width//2 : row-patch_width//2 + patch_width, col-patch_height//2 : col-patch_height//2 + patch_height
            ]
            regions_mask_sample = regions_mask[
                row-patch_width//2 : row-patch_width//2 + patch_width, col-patch_height//2 : col-patch_height//2 + patch_height
            ]

            gr_aug_sample, gt_aug_sample, regions_mask_aug_sample, applied_augmentations = apply_random_augmentations(gr_sample, gt_sample, regions_mask_sample, augmentation_types, patch_width, patch_height)
            
            current_rate_annotated_pixels = np.sum(gt_aug_sample == 1) / (patch_height*patch_width)
            if current_rate_annotated_pixels >= min_rate_annotated_pixels or tries > MAX_TRIES:
                gr_chunks.append(gr_aug_sample)
                gt_chunks.append(gt_aug_sample)
                num_samples+=1
                tries = 0
            else:
                tries+=1
    else:
        print("No annotated pixels found...")
        x_coords = [
            random.randint(0, gr.shape[0]-patch_width-1) for _ in range(batch_size)
        ]

        y_coords = [
            random.randint(0, gr.shape[1]-patch_height-1) for _ in range(batch_size)
        ]

        for i in range(batch_size):
            row = x_coords[i]
            col = y_coords[i]
            
            row = max(patch_width//2, row)
            col = max(patch_height//2, col)

            row = min(gr.shape[0]-patch_width//2, row)
            col = min(gr.shape[1]-patch_height//2, col)
            
            appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks)
    

def apply_mask(gt_img, regions_mask=None):
    if regions_mask is not None:
        masked = np.logical_and(gt_img, regions_mask)*1
        return masked
    else:
        return gt_img

def calculate_mask(gt, window_w, window_h, nb_sequential_patches = -1, batch_size=-1):
    ROWS = gt.shape[0]
    COLS = gt.shape[1]
    
    min_rate_annotated_pixels = 0.0025

    mask = np.zeros((ROWS, COLS))

    patch_counter = 0
    
    for row in range(window_w//2, ROWS+window_w//2-1, window_w):
        for col in range(window_h//2, COLS+window_h//2-1, window_h):
            row = min(row, ROWS-window_w//2)
            col = min(col, COLS-window_h//2)
            
            gt_sample = gt[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h]
            
            
            if (np.sum(gt_sample == 1) > batch_size):
                current_rate_annotated_pixels = np.sum(gt_sample == 1) / (window_h*window_w)
                            
                if nb_sequential_patches == -1 or current_rate_annotated_pixels >= min_rate_annotated_pixels:
                    
                    mask[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h] = 1
                    patch_counter += 1

                    if nb_sequential_patches != -1 and patch_counter >=nb_sequential_patches:
                        return mask, patch_counter

    return mask, patch_counter


def get_gt_image_and_regions(gt_path_file, nb_sequential_patches, window_w, window_h, batch_size):
    gt_img = (utilIO.load_gt_image(gt_path_file)[:,:,3] > 128)*1 #Annotations are in alpha channel

    regions_mask, n_patches = calculate_mask(gt_img, window_w, window_h, nb_sequential_patches, batch_size)
    gt_img = apply_mask(gt_img, regions_mask=regions_mask)

    return gt_img, regions_mask, n_patches


def normalize_image(img):
    return (255.-img) / 255.

def get_image_with_gt(page_src, page_gt, nb_sequential_patches, window_w, window_h, batch_size, with_mask=False):

    gt, regions_mask, n_annotated_patches_real = get_gt_image_and_regions(page_gt, nb_sequential_patches, window_w, window_h, batch_size)
    gr = utilIO.load_src_image(page_src)
    gr = normalize_image(gr)

    if with_mask:
        #Deactivate the training process for pixels outside the region mask
        l = np.where((regions_mask == 0))
        gr[l] = utilConst.kPIXEL_VALUE_FOR_MASKING

    return gr, gt, regions_mask, n_annotated_patches_real


def dump_image_with_size(gr, gt, regions_mask, width_out, height_out):

    ROWS=gr.shape[0]
    COLS=gr.shape[1]

    center_w = ROWS // 2
    center_h = COLS // 2

    if (len(gr.shape) == 3):
        gr_new = np.ones((width_out, height_out, gr.shape[2]))*(-1)
    else:
        gr_new = np.ones((width_out, height_out))*(-1)
    gt_new = np.zeros((width_out, height_out))
    regions_mask_new = np.zeros((width_out, height_out))

    rows_to_copy = min(ROWS, width_out)
    cols_to_copy = min(COLS, height_out)

    center_w = ROWS // 2
    center_h = COLS // 2
    center_w_new = width_out // 2
    center_h_new = height_out // 2

    gr_new[center_w_new-rows_to_copy//2:center_w_new-rows_to_copy//2 + rows_to_copy, center_h_new-cols_to_copy//2:center_h_new-cols_to_copy//2 + cols_to_copy] = gr[center_w-rows_to_copy//2:center_w-rows_to_copy//2+rows_to_copy, center_h-cols_to_copy//2:center_h-cols_to_copy//2+cols_to_copy]
    if gt is not None:
        gt_new[center_w_new-rows_to_copy//2:center_w_new-rows_to_copy//2 + rows_to_copy, center_h_new-cols_to_copy//2:center_h_new-cols_to_copy//2 + cols_to_copy] = gt[center_w-rows_to_copy//2:center_w-rows_to_copy//2+rows_to_copy, center_h-cols_to_copy//2:center_h-cols_to_copy//2+cols_to_copy]
    regions_mask_new[center_w_new-rows_to_copy//2:center_w_new-rows_to_copy//2 + rows_to_copy, center_h_new-cols_to_copy//2:center_h_new-cols_to_copy//2 + cols_to_copy] = regions_mask[center_w-rows_to_copy//2:center_w-rows_to_copy//2+rows_to_copy, center_h-cols_to_copy//2:center_h-cols_to_copy//2+cols_to_copy]

    return gr_new, gt_new, regions_mask_new


def apply_random_augmentations(gr, gt, regions_mask, augmentation_types, width_out, height_out):

    gr_aug = gr
    gt_aug = gt
    regions_mask_aug = regions_mask
    applied_augmentations = []

    augmentation_types_aux = augmentation_types
    if "random" in augmentation_types:
        augmentation_types_aux = [item for item in augmentation_types if item != "random"]
        if len(augmentation_types_aux) == 0:
            augmentation_types_aux.append("none")
        
    random.shuffle(augmentation_types_aux)
    
    for augmentation_type in augmentation_types_aux:
        activate_augmentation = random.randint(0, 1) == 1

        if activate_augmentation:
            gr_aug, gt_aug, regions_mask_aug, type_augmentation_out = apply_augmentation(gr_aug, gt_aug, regions_mask_aug, augmentation_type)
            applied_augmentations.append(type_augmentation_out)

    gr_new, gt_new, regions_mask_new =  dump_image_with_size(gr_aug, gt_aug, regions_mask_aug, width_out, height_out)
    return gr_new, gt_new, regions_mask_new, applied_augmentations



def getRandomSamples(page, batch_size, nb_annotated_patches, window_w, window_h, augmentation_types):
    gr_chunks = []
    gt_chunks = []
 
    gr, gt, regions_mask, n_annotated_patches_real = get_image_with_gt(page[0], page[1], nb_annotated_patches, window_w, window_h, batch_size, True)
    
    while len(gr_chunks) < batch_size:
        extractRandomSamplesClass(gr, gt, window_w, window_h, 1, gr_chunks, gt_chunks, regions_mask, augmentation_types)

    gr_chunks_arr = np.array(gr_chunks)
    gt_chunks_arr = np.array(gt_chunks)
    gt_chunks_arr = np.reshape(gt_chunks_arr, (gt_chunks_arr.shape[0], gt_chunks_arr.shape[1], gt_chunks_arr.shape[2], 1))
    # convert gr_chunks and gt_chunks to the numpy arrays that are yield below    

    yield gr_chunks_arr, gt_chunks_arr



def getSequentialSamples(gr, gt, regions_mask, idx_patch, batch_size, n_annotated_patches_real, nb_annotated_patches, window_w, window_h, augmentation_types):
    
    #print("Annotated:")
    #print(n_annotated_patches_real)
    patch_counter = 0
    gr_chunks = []
    gt_chunks = []    
    patch_counter = idx_patch
    while len(gr_chunks) < batch_size and patch_counter < min(n_annotated_patches_real, nb_annotated_patches):
        patch_counter = extractSequentialSamplesClass(gr, gt, window_w, window_h, 1, patch_counter, gr_chunks, gt_chunks, regions_mask, augmentation_types)
        if len(gr_chunks) == 0:
            print ("Is none")
            return None
        
    gr_chunks_arr = np.array(gr_chunks)
    gt_chunks_arr = np.array(gt_chunks)
    gt_chunks_arr = np.reshape(gt_chunks_arr, (gt_chunks_arr.shape[0], gt_chunks_arr.shape[1], gt_chunks_arr.shape[2], 1))
    # convert gr_chunks and gt_chunks to the numpy arrays that are yield below    
    return gr_chunks_arr, gt_chunks_arr

def get_number_annotated_patches(page, window_w, window_h, number_patches=-1):
    if type(page) is tuple:
        gr, gt, regions_mask, n_annotated_patches_real_total = get_image_with_gt(page[0], page[1], number_patches, window_w, window_h, 1, True)
    else:
        n_annotated_patches_real_total = 0
        for p in page:
            gr, gt, regions_mask, n_annotated_patches_real = get_image_with_gt(p[0], p[1], number_patches, window_w, window_h, 1, True)
            n_annotated_patches_real_total += n_annotated_patches_real
    return n_annotated_patches_real_total

def create_generator(data_pages, no_mask, batch_size, window_shape, nb_patches, nb_annotated_patches, augmentation_types):
    if no_mask is None or no_mask == False:
        using_mask = True
    else:
        using_mask = False 
    while(True):
        #print("Shuffle training data...")
        random.shuffle(data_pages)
        #print("Done")

        for page in data_pages:
            if utilConst.AUGMENTATION_RANDOM in augmentation_types:
                assert(nb_patches != -1)
                yield from getRandomSamples(page, min(batch_size, nb_patches), nb_annotated_patches, window_shape[0], window_shape[1], augmentation_types)
            else:
                assert(nb_annotated_patches == nb_patches)
                real_patches = get_number_annotated_patches(page, window_shape[0], window_shape[1], nb_annotated_patches)
                if nb_annotated_patches == -1:
                    nb_annotated_patches_real = real_patches
                    np_patches_real = real_patches
                else:
                    nb_annotated_patches_real = nb_annotated_patches
                    np_patches_real = nb_patches
                gr, gt, regions_mask, n_annotated_patches_real = get_image_with_gt(page[0], page[1], nb_annotated_patches_real, window_shape[0], window_shape[1], batch_size, using_mask)
                idx_patch = 0
                while idx_patch < min(n_annotated_patches_real, nb_annotated_patches_real):    
                    samples = getSequentialSamples(gr, gt, regions_mask, idx_patch, min(batch_size, real_patches), n_annotated_patches_real, n_annotated_patches_real, window_shape[0], window_shape[1], augmentation_types)
                    if samples is not None:
                        idx_patch += len(samples[0])
                        yield samples[0], samples[1]
                    else:
                        idx_patch = min(n_annotated_patches_real, nb_annotated_patches)
                        


def __run_validations(pred, gt):
    assert(isinstance(pred, np.ndarray))
    assert(isinstance(gt, np.ndarray))

    assert(np.issubdtype(pred.dtype.type, np.bool_))
    assert(np.issubdtype(gt.dtype.type, np.bool_))

    assert(len(pred) == len(gt))
    assert(pred.shape[0]==gt.shape[0])


def __calculate_metrics(prediction, gt):
    __run_validations(prediction, gt)

    not_prediction = np.logical_not(prediction)
    not_gt = np.logical_not(gt)

    tp = np.logical_and(prediction, gt)
    tn = np.logical_and(not_prediction, not_gt)
    fp = np.logical_and(prediction, not_gt)
    fn = np.logical_and(not_prediction, gt)

    tp = (tp.astype('int32')).sum()
    tn = (tn.astype('int32')).sum()
    fp = (fp.astype('int32')).sum()
    fn = (fn.astype('int32')).sum()

    epsilon = 0.00001
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    fm = 2 * (precision * recall) / (precision + recall + epsilon)
    specificity = tn / (tn + fp + epsilon)

    gt = gt.astype('int32')
    prediction = prediction.astype('int32')

    difference = np.absolute(prediction - gt)
    totalSize = np.prod(gt.shape)
    error = float(difference.sum()) / float(totalSize)

    return {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn,
            'error':error, 'accuracy':accuracy, 'precision':precision,
            'recall':recall, 'fm':fm, 'specificity':specificity}



#imutils version adapted to RGB
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    if len(image.shape) >= 3:
        image_out = np.zeros([nH, nW,image.shape[2]])
        for channel in range(image.shape[2]):
            
            image_rotated = cv2.warpAffine(image[:,:,channel], M, (nW, nH))
            image_out[:,:,channel] = image_rotated        
    else:
        image_out = cv2.warpAffine(image, M, (nW, nH))
    
    return image_out

#https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
def zoom_at(img, zoom, coord=None):
    """
    Simple image zooming without boundary checking.
    Centered at "coord", if given, else the image center.

    img: numpy.ndarray of shape (h,w,:)
    zoom: float
    coord: (float, float)
    """
    # Translate to zoomed coordinates
    h, w, _ = [ zoom * i for i in img.shape ]
    
    if coord is None: cx, cy = w/2, h/2
    else: cx, cy = [ zoom*c for c in coord ]
    
    img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
    img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
               int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
               : ]
    
    return img

def apply_augmentation(x_image, y_image, regions_mask, type_augmentation, value_augmentation=None):
    x_image_out = None
    y_image_out = None
    type_augmentation_out = None
    
    if y_image is None:
        y_image_float = None
    else:
        y_image_float = y_image.astype(np.float64)
    regions_mask_float = regions_mask.astype(np.float64)
    regions_mask_out = regions_mask_float 
    
    if type_augmentation == utilConst.AUGMENTATION_NONE:
        x_image_out = x_image
        y_image_out = y_image
        regions_mask_out = regions_mask
        type_augmentation_out = (type_augmentation, 0)
    elif type_augmentation == utilConst.AUGMENTATION_FLIPH:
        x_image_out = cv2.flip(x_image, 1)
        if y_image is not None:
            y_image_out = cv2.flip(y_image, 1)
        regions_mask_out = cv2.flip(regions_mask, 1)
        type_augmentation_out = (type_augmentation, 1)
        
    elif type_augmentation == utilConst.AUGMENTATION_FLIPV:
        x_image_out = cv2.flip(x_image, -1)
        if y_image is not None:
            y_image_out = cv2.flip(y_image, -1)
        regions_mask_out = cv2.flip(regions_mask, -1)
        type_augmentation_out = (type_augmentation, -1)
    elif type_augmentation == utilConst.AUGMENTATION_ROTATION:
        if value_augmentation is None:
            angle = random.uniform(-5, 5)
        else:
            angle = value_augmentation
        x_image_out = rotate_bound(x_image, angle)
        if y_image is not None:
            y_image_out = rotate_bound(y_image_float, angle)
        regions_mask_out = (rotate_bound(regions_mask_float, angle) > 0) * 1
        if y_image is not None:
            y_image_out = apply_mask(y_image_out, regions_mask_out)

        type_augmentation_out = (type_augmentation, angle)
        
    elif type_augmentation == utilConst.AUGMENTATION_SCALE:
        if value_augmentation is None:
            zoom_factor = random.uniform(0.90, 1.10)
        else:
            zoom_factor = value_augmentation
        ROWS = x_image.shape[0]
        COLS = x_image.shape[1]

        x_image_out = cv2.resize(x_image, None, fx=zoom_factor, fy=zoom_factor)
        if y_image is not None:
            y_image_out = cv2.resize(y_image_float, None, fx=zoom_factor, fy=zoom_factor)
        regions_mask_out = cv2.resize(regions_mask_float, None, fx=zoom_factor, fy=zoom_factor)
        if y_image is not None:
            y_image_out = apply_mask(y_image_out, regions_mask_out)
        type_augmentation_out = (type_augmentation, zoom_factor)
        
    elif type_augmentation == utilConst.AUGMENTATION_DROPOUT:
        assert (False)
        
    regions_mask_out = (regions_mask_out>0.5)*1
    l = np.where((regions_mask_out == 0))
    x_image_out[l] = utilConst.kPIXEL_VALUE_FOR_MASKING
    y_image_out[l] = 0
                
    return x_image_out, y_image_out, regions_mask_out, type_augmentation_out

#------------------------------------------------------------------------------
def run_test(y_pred, y_gt, threshold=.5):
    prediction = y_pred.copy()
    gt = y_gt.copy()

    if threshold is not None:
        prediction = (prediction > threshold)
    else:
        prediction = (prediction > 0.5)

    gt = gt > 0.5

    r = __calculate_metrics(prediction, gt)

    return r



def get_best_threshold(y_pred, y_test, verbose=1, args_th=None):
    best_fm = -1
    best_th = -1
    prec = 0.
    recall = 0.
    
    if args_th is None:
        for i in range(1, 10, 1):
            th = float(i) / 10.0
            #print('Threshold:', th)
            results = run_test(y_pred, y_test, threshold=th)
            fm = results['fm']
            if fm > best_fm:
                best_fm = fm
                best_th = th
                prec = results['precision']
                recall = results['recall']
        if verbose:
            print('Best threshold:', best_th)
            print("Best Fm: %.4f " % best_fm, 
                    "P: %.3f " % prec,
                    "R: %.3f " % recall)
    else:
        results = run_test(y_pred, y_test, threshold=args_th)
        best_fm = results['fm']
        best_th = args_th
        prec = results['precision']
        recall = results['recall']
        if verbose:
            print('Threshold:', best_th)
            print("Fm: %.4f " % best_fm, 
                    "P: %.3f " % prec,
                    "R: %.3f " % recall)

    return best_fm, best_th, prec, recall


def compute_best_threshold(path_model, val_data, batch_size, window_shape, nb_annotated_patches=-1, threshold=None, with_masked_input=True):
    model = tf.keras.models.load_model(path_model)
    window_w = window_shape[0]
    window_h = window_shape[1]
    
    predictions = np.array(list())
    gts = np.array(list())
    idx = 0
    dict_predictions = {}
    for page_test in val_data:
        page_src = page_test[0]
        page_gt = page_test[1]

        idx+=1
        print("Processing..." + str(idx) + "/" + str(len(val_data)) + ": " + page_src)
        
        gr, gt, region_mask, n_annotated_patches_real = get_image_with_gt(page_src, page_gt, nb_annotated_patches, window_w, window_h, batch_size, with_masked_input)
        
        prediction = predict_image(model, gr, -1, window_shape)
        coords_with_annotations = np.where((region_mask.flatten())!=0)
        
        dict_predictions[page_src]  = prediction
        
        predictions = np.concatenate((predictions, (prediction.flatten())[coords_with_annotations]))
        gts = np.concatenate((gts, (gt.flatten())[coords_with_annotations]))
       
    #predictions = np.array(predictions)
    #gts = np.array(gts)
    best_fm, best_th, prec, recall = get_best_threshold(predictions, gts, verbose=1, args_th=threshold)
        
    return best_fm, best_th, prec, recall, dict_predictions


def compute_metrics(config, path_model, test_data, batch_size, window_shape, nb_annotated_patches=-1, threshold=None, with_masked_input=True):
    import CNNmodel
    no_mask = not with_masked_input
    model = CNNmodel.get_model(window_shape, no_mask, config.n_la, config.nb_fil, config.ker, dropout=config.drop, stride=2)
    model.load_weights(path_model)
    
    #model = tf.keras.models.load_model(path_model)
    window_w = window_shape[0]
    window_h = window_shape[1]
 
    
    idx = 0
    dict_predictions = {}
    for page_test in test_data:
        page_src = page_test[0]
        page_gt = page_test[1]

        idx+=1
        print("Processing..." + str(idx) + "/" + str(len(test_data)) + ": " + page_src)
        
        gr, gt, _, _ = get_image_with_gt(page_src, page_gt, nb_annotated_patches, window_w, window_h, batch_size, False)
        gt=gt>0.5
        prediction_matrix = predict_image(model, gr, -1, window_shape)

        
        path_result = path_model.replace("models/modelCNN/", "tests/").replace(".h5", "/") + page_test[0].replace("datasets/", "")
        utilIO.saveImage((gt)*255, path_result + "_gt.png") 
        utilIO.saveImage((gr)*255, path_result + "_gr.png")
        utilIO.saveImage((prediction_matrix)*255, path_result + "_pred.png")
        utilIO.saveImage((prediction_matrix>threshold)*255, path_result + "_pred_th.png")
        
        gr=None
        gt=None

        if utilConst.KEY_RESULT not in dict_predictions:
            dict_predictions[utilConst.KEY_RESULT] = {}    
        dict_predictions[utilConst.KEY_RESULT][page_src] = {}
        dict_predictions[utilConst.KEY_RESULT][page_src][0] = prediction_matrix

    dict_results = {}
    
    predictions = np.array(list())
    gts = np.array(list())
    for page_test in test_data:
        page_src = page_test[0]
        page_gt = page_test[1]
        
        gr, gt, _, _ = get_image_with_gt(page_src, page_gt, nb_annotated_patches, window_w, window_h, batch_size, with_masked_input)
        coords_with_annotations = np.where((dict_predictions[utilConst.KEY_RESULT][page_src][0].flatten())!=utilConst.kPIXEL_VALUE_FOR_MASKING)
        predictions = np.concatenate((predictions, (dict_predictions[utilConst.KEY_RESULT][page_src][0].flatten())[coords_with_annotations]))
        gts = np.concatenate((gts, (gt.flatten())[coords_with_annotations]))
    if len(predictions) != 0 and len(gts) != 0:
        best_fm, best_th, prec, recall = get_best_threshold(predictions, gts, verbose=1, args_th=threshold)
        if utilConst.KEY_RESULT not in dict_results:
            dict_results[utilConst.KEY_RESULT] = {}    
        dict_results[utilConst.KEY_RESULT][0] = (best_fm, prec, recall)


    return dict_results, dict_predictions


def predict_image(model, gr_norm, nb_sequential_patches, window_shape):
    
    window_w = window_shape[0]
    window_h = window_shape[1]
        
    ROWS = gr_norm.shape[0]
    COLS = gr_norm.shape[1]

    prediction = np.ones((ROWS, COLS))*(-1)
    
    margin = 10
    patch_counter = 0

    
    for row in range(window_w//2, ROWS+window_w//3-1, window_w//3):
        for col in range(window_h//2, COLS+window_h//3-1, window_h//3):
            row = min(row, ROWS-window_w//2)
            col = min(col, COLS-window_h//2)
            
            patch_gr = gr_norm[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h]
            list_patches_batch = []
            list_patches_batch.append(patch_gr)
            list_masks = []
            list_masks.append(None)
            
            patch_gr_arr = np.array(list_patches_batch)
            
            predicted_patches = model.predict(patch_gr_arr, verbose=0)[:,:,:,0]
            prediction[row-window_w//2+margin:row-window_w//2+window_w-margin, col-window_h//2+margin:col-window_h//2+window_h-margin, 0] = np.maximum(prediction[row-window_w//2+margin:row-window_w//2+window_w-margin, col-window_h//2+margin:col-window_h//2+window_h-margin,0], predicted_patches[0,margin:-margin,margin:-margin])

            
            predicted_patch = predicted_patches[margin:-margin,margin:-margin]
            regions_mask_aug_sample = list_masks
            prediction_correct = predicted_patch
            regions_mask_correct = regions_mask_aug_sample
            
            l = np.where((regions_mask_correct == 0))
            prediction_correct[l] = utilConst.kPIXEL_VALUE_FOR_MASKING

            prediction[row-window_w//2+margin:row-window_w//2+window_w-margin, col-window_h//2+margin:col-window_h//2+window_h-margin] = prediction_correct
            
            patch_counter+=1
            if (nb_sequential_patches != -1 and patch_counter >=nb_sequential_patches*2) or nb_sequential_patches == 1:
                return prediction

    return prediction





def predict_image(model, gr_norm, nb_sequential_patches, window_shape):
    

    window_w = window_shape[0]
    window_h = window_shape[1]
        
    ROWS = gr_norm.shape[0]
    COLS = gr_norm.shape[1]

    prediction = np.zeros((ROWS, COLS))
    
    margin = 5
    patch_counter = 0

    for row in range(window_w//2, ROWS+window_w//2-1, window_w//2):
        for col in range(window_h//2, COLS+window_h//2-1, window_h//2):
            row = min(row, ROWS-window_w//2)
            col = min(col, COLS-window_h//2)
            
            patch_gr = gr_norm[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h]
            patch_gr_arr = np.array(patch_gr)
            patch_gr_arr = np.reshape(patch_gr_arr, (1, patch_gr_arr.shape[0], patch_gr_arr.shape[1], patch_gr_arr.shape[2]))
            
            predicted_patch = model.predict(patch_gr_arr, verbose=0)[0,:,:,0]
            
            prediction[row-window_w//2+margin:row-window_w//2+window_w-margin, col-window_h//2+margin:col-window_h//2+window_h-margin] = np.maximum(prediction[row-window_w//2+margin:row-window_w//2+window_w-margin, col-window_h//2+margin:col-window_h//2+window_h-margin], predicted_patch[margin:-margin,margin:-margin])
            patch_counter+=1
            if (nb_sequential_patches != -1 and patch_counter >=nb_sequential_patches*2) or nb_sequential_patches == 1:
                return prediction

    return prediction


def test_model(config, path_model, test_data, window_shape, threshold, with_masked_input):
    dict_results, dict_predictions = compute_metrics(config=config, path_model=path_model, test_data=test_data, batch_size=1, window_shape=window_shape, nb_annotated_patches=-1, threshold=threshold, with_masked_input=with_masked_input)
    
    pathfolder_result = path_model.replace(".h5", "/").replace("models/", "results/")
    pathfolder_result_bin = path_model.replace(".h5", "/").replace("models/", "results/bin/")

        
    return dict_results

    
    
    