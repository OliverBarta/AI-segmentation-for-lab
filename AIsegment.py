import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Cropping2D
from scipy.io import loadmat

#U-Net architecture for segmentation (actually training part\model)
def unet(input_size=(380, 308, 1)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    # Decoder
    u4 = Conv2DTranspose(32, 2, strides=2, padding='same')(c3)

    # Crop c2 if needed
    if u4.shape[1] != c2.shape[1] or u4.shape[2] != c2.shape[2]:
        crop_h = c2.shape[1] - u4.shape[1]
        crop_w = c2.shape[2] - u4.shape[2]
        c2 = Cropping2D(cropping=((0, crop_h), (0, crop_w)))(c2)

    u4 = concatenate([u4, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u4)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u5 = Conv2DTranspose(16, 2, strides=2, padding='same')(c4)

    # Crop c1 if needed
    if u5.shape[1] != c1.shape[1] or u5.shape[2] != c1.shape[2]:
        crop_h = c1.shape[1] - u5.shape[1]
        crop_w = c1.shape[2] - u5.shape[2]
        c1 = Cropping2D(cropping=((0, crop_h), (0, crop_w)))(c1)

    u5 = concatenate([u5, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

#Loads your training data, the number thats in the thousands below is how many frames are actually used to train on. 
#If the number is higher then your ram can sustain, the program will crash.
def load_data(image_dir, mask_dir, img_size=(308, 380)):
    images = []
    masks = []
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)
    num_frames_added = 0
    count = 1
    for img_file, mask_file in zip(image_files, mask_files):
        if num_frames_added >= 42000:
            break
        #loads image file (ultrasound video)
        b_mode = loadmat(os.path.join(image_dir,img_file))
        b_mode_f = b_mode["b_mode_f"]   #adjust if key is different (it shouldn't be if you ran the "process_mat_files.py script")

        #Loads mask file
        mask_data = loadmat(os.path.join(mask_dir, mask_file))
        mask_array = mask_data['mask']  #adjust if key is different (it shouldn't be if you ran the "process_mat_files.py script")

        #print to show progress and for user to know what has been loaded for training
        print(count,img_file,"mask shape: ", mask_array.shape, "img shape: ",b_mode_f.shape)
        count += 1

        #Takes only 1267 frames from every ultrasound video so that every video can get processed
        num_frames = 1267#if this crashes change this number to 1213

        for i in range(num_frames):
            frame = b_mode_f[i]
            mask = mask_array[i]

            # Resize and normalize
            frame_resized = cv2.resize(frame, img_size)
            mask_resized = cv2.resize(mask, img_size)

            frame_resized = frame_resized / 255.0
            mask_resized = mask_resized / 255.0

            frame_resized = np.expand_dims(frame_resized, axis=-1)
            mask_resized = np.expand_dims(mask_resized, axis=-1)

            images.append(frame_resized)
            masks.append(mask_resized)
            num_frames_added += 1
    
    print("Frames used:",num_frames_added)
    return np.array(images), np.array(masks)


#initializes model training
image_dir = 'Segmentation\Matched_processed_img'   #processed and matched ultrasound frames
mask_dir = 'Segmentation\Matched_processed_mask'   #processed and matched corresponding mask frames
X_train, Y_train = load_data(image_dir, mask_dir)

model = unet(input_size=(380, 308, 1))
model.fit(X_train, Y_train, batch_size=16, epochs=20, validation_split=0.1)#edits model to fit this data

#saves model so you can use it without having to run this again. (to use model run seperate script)
model.save("AI_segmentation_unet_6_1267_frames_per.h5")

#The first model is named "AI_segmentation_unet.h5"
#The second model is named "AI_segmentation_unet_2_667_frames_per.h5"
#the third model is named "AI_segmentation_unet_3_1267_frames_per.h5"
#the fourth model is named "AI_segmentation_unet_4_1267_frames_double_filters.h5"
#the fifth model is named "AI_segmentation_unet_5_1267_frames_double_filters.h5"
#the sixth model is named "AI_segmentation_unet_6_1267_frames_per.h5"
