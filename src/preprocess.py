import numpy as np
from astropy.io import fits
import cv2 as cv
import numpy as np
import copy
import os
import math

def _write_npy(image, output_destination):
    if not os.path.isfile(output_destination + '.npy'):
        np.save(output_destination + '.npy', image)

def _write_png(image, output_destination):
    if not os.path.isfile(output_destination + '.png'):
        cv.imwrite(output_destination + '.png', image)

def convert_fits(files_path, output_path, classes, split=True, output_format='npy'):

    class_counts = dict.fromkeys(classes, 0)

    increment = len(classes)

    for index, file_path in enumerate(files_path):
        ### Categorizing and counting the data. We take 3 slices from each FITS, so we increment the count +3
        classname = ""
        for classname in classes:
            if classname in file_path:
                class_counts[classname] += 3

                export_name = output_path + str(index) + "_" + str(classname)


                raw = fits.getdata(file_path)

                if split:
                    half = math.floor(raw.shape[0] / 2)
                    img_f = copy.deepcopy(raw[0,:,:])
                    img_m = copy.deepcopy(raw[half,:,:])
                    img_l = copy.deepcopy(raw[raw.shape[0]-1,:,:])

                    ret, img_first = cv.threshold(img_f, 127, 255, cv.THRESH_BINARY)
                    ret, img_middle = cv.threshold(img_m, 127, 255, cv.THRESH_BINARY)
                    ret, img_last = cv.threshold(img_l, 127, 255, cv.THRESH_BINARY)

                    if output_format=='npy':
                        _write_npy(img_first, output_path + export_name + "0")
                        _write_npy(img_middle, output_path + export_name + str(half))
                        _write_npy(img_last, output_path + export_name + str(raw.shape[0]-1))
                    elif output_path=='png':
                        _write_png(img_first, output_path + export_name + "0")
                        _write_png(img_middle, output_path + export_name + str(half))
                        _write_png(img_last, output_path + export_name + str(raw.shape[0]-1))

                else:
                    for i in range(raw.shape[2]):
                        ret, img= cv.threshold(raw[i,:,:], 127, 255, cv.THRESH_BINARY)
                        if output_format=='npy':
                            _write_npy(img, output_path + export_name + str(i))
                        elif output_path=='png':
                            _write_png(img, output_path + export_name + str(i))

            break # break classname for loop


def flatten_images(files_path, output_path, classes, count_max, crop_size=[300,200]):
    column_names = []
    row_length = crop_size[0] * crop_size[1]
    for i in range(0,row_length):
        column_names.append('feature_' + str(i))

    df = pd.DataFrame(columns=column_names)
    df.insert(row_length, 'class', "", True)


    files = os.listdir(files_path)

    class_counts = dict.fromkeys(classes, 0)

    for i, filename in enumerate(files):
        for classname in classes:
            if classname in filename and class_counts[classname] <= count_max:
                if class_counts[classname] == count_max:
                    continue
                class_counts[classname] += 1

            image = np.load(files_path + filename)
            image_crop = image[0:crop_size[0], 0:crop_size[1]]

            image_flat = image_crop.flatten()
            image_flat = image_flat/255
    
            data = np.append(image_flat, classname)

            if len(data) == 60001:
                df.loc[len(df)] = data
            else:
                print(filename)
                print(len(data))
    

            if i == count_max:
                export_fname = flat_folder + 'all_data_' + str(i - 4) + '-' + str(i) + '.csv.gz'
                if os.path.isfile(export_fname):
                    continue
                df.to_csv(flat_folder + 'all_data_' + str(i - 4) + '-' + str(i) + '.csv.gz', index=False, compression='infer')

            if i % 10 == 0 and i != 0:
                print(i - 10, '-', i)
                export_fname = flat_folder + 'all_data_' + str(i - 10) + '-' + str(i) + '.csv.gz'
                if os.path.isfile(export_fname):
                    continue

                df.to_csv(flat_folder + 'all_data_' + str(i - 10) + '-' + str(i) + '.csv.gz', index=False, compression='infer')


                df = df.iloc[0:0]
                df = pd.DataFrame(columns=column_names)
                df.insert(60000, 'class', "", True)
                continue
    
