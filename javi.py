
import random
import numpy as np
import cv2

kPIXEL_VALUE_FOR_MASKING = -1
kNUMBER_CHANNELS = 3

AUGMENTATION_NONE = "none"
AUGMENTATION_FLIPH = "flipH"
AUGMENTATION_FLIPV = "flipV"
AUGMENTATION_ROTATION = "rot"
AUGMENTATION_SCALE = "scale"
AUGMENTATION_DROPOUT = "drop"
AUGMENTATION_RANDOM = "random"


def load_gt_image(path_file, regions_mask=None):
    file_img = cv2.imread(path_file, cv2.IMREAD_UNCHANGED,)  # 4-channel
    if file_img is None : 
        raise Exception(
            'It is not possible to load the image\n'
            "Path: " + str(path_file)
        )
    
    return file_img

def load_src_image(path_file, mode=cv2.IMREAD_COLOR):

    file_img = cv2.imread(path_file, mode)
    if file_img is None : 
        raise Exception(
            'It is not possible to load the image\n'
            "Path: " + str(path_file)
        )

    return file_img


def normalize_image(img):
    return (255.-img) / 255.


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
    gt_img = (load_gt_image(gt_path_file)[:,:,3] > 128)*1 #Annotations are in alpha channel

    regions_mask, n_patches = calculate_mask(gt_img, window_w, window_h, nb_sequential_patches, batch_size)
    gt_img = apply_mask(gt_img, regions_mask=regions_mask)

    return gt_img, regions_mask, n_patches

def get_image_with_gt(page_src, page_gt, nb_sequential_patches, window_w, window_h, batch_size, with_mask=False):

    gt, regions_mask, n_annotated_patches_real = get_gt_image_and_regions(page_gt, nb_sequential_patches, window_w, window_h, batch_size)
    gr = load_src_image(page_src)
    gr = normalize_image(gr)

    if with_mask:
        #Deactivate the training process for pixels outside the region mask
        l = np.where((regions_mask == 0))
        gr[l] = kPIXEL_VALUE_FOR_MASKING

    return gr, gt, regions_mask, n_annotated_patches_real




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
    
    if type_augmentation == AUGMENTATION_NONE:
        x_image_out = x_image
        y_image_out = y_image
        regions_mask_out = regions_mask
        type_augmentation_out = (type_augmentation, 0)
    elif type_augmentation == AUGMENTATION_FLIPH:
        x_image_out = cv2.flip(x_image, 1)
        if y_image is not None:
            y_image_out = cv2.flip(y_image, 1)
        regions_mask_out = cv2.flip(regions_mask, 1)
        type_augmentation_out = (type_augmentation, 1)
        
    elif type_augmentation == AUGMENTATION_FLIPV:
        x_image_out = cv2.flip(x_image, -1)
        if y_image is not None:
            y_image_out = cv2.flip(y_image, -1)
        regions_mask_out = cv2.flip(regions_mask, -1)
        type_augmentation_out = (type_augmentation, -1)
    elif type_augmentation == AUGMENTATION_ROTATION:
        if value_augmentation is None:
            angle = random.uniform(-45, 45)
        else:
            angle = value_augmentation
        x_image_out = rotate_bound(x_image, angle)
        if y_image is not None:
            y_image_out = rotate_bound(y_image_float, angle)
        regions_mask_out = (rotate_bound(regions_mask_float, angle) > 0) * 1
        if y_image is not None:
            y_image_out = apply_mask(y_image_out, regions_mask_out)
        
        l = np.where((regions_mask_out == 0))
        x_image_out[l] = kPIXEL_VALUE_FOR_MASKING

        type_augmentation_out = (type_augmentation, angle)
        
    elif type_augmentation == AUGMENTATION_SCALE:
        if value_augmentation is None:
            zoom_factor = random.uniform(0.80, 1.20)
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

        l = np.where((regions_mask_out == 0))
        x_image_out[l] = kPIXEL_VALUE_FOR_MASKING
        
    elif type_augmentation == AUGMENTATION_DROPOUT:
        assert (False)
                
    return x_image_out, y_image_out, regions_mask_out, type_augmentation_out




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
    if AUGMENTATION_RANDOM in augmentation_types:
        augmentation_types_aux = [item for item in augmentation_types if item != AUGMENTATION_RANDOM]
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




def appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks):
    gr_sample = gr[
            row : row + patch_height, col : col + patch_width
        ]  # Greyscale image
    gt_sample = gt[
        row : row + patch_height, col : col + patch_width
    ]  # Ground truth
    gr_chunks.append(gr_sample)
    gt_chunks.append(gt_sample)


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

            row = max(patch_width//2+1, row)
            col = max(patch_height//2+1, col)

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




def create_generator(data_pages, batch_size, window_shape, nb_patches, nb_annotated_patches, augmentation_types):

    while(True):
        #print("Shuffle training data...")
        random.shuffle(data_pages)
        #print("Done")

        for page in data_pages:
            assert(nb_patches != -1)
            yield from getRandomSamples(page, min(batch_size, nb_patches), nb_annotated_patches, window_shape[0], window_shape[1], augmentation_types)
        
                        
