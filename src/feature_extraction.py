def extract_hog_features(image):
    from skimage.feature import hog
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=True)
    return hog_features[0]