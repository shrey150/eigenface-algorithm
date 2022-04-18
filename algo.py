import matplotlib.pyplot as plt
import numpy as np
import os
import random

# display faces with labels for predictions
def plot_portraits(images, titles, h, w, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

# perform principal component analysis
def pca(X, n_pc):
    # get the mean of each image and center its vector on the origin
    mean = np.mean(X, axis=0)
    centered_data = X-mean

    # perform singular value decomposition
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:,:n_pc]*S[:n_pc]
    
    return projected, components, mean, centered_data

##############################################################################

DIR_NAME = 'lfwcrop_grey/faces'
n_components = 50
TRAIN_SIZE = 800
TEST_SIZE = 200

# load in images and filenames
celebrity_photos = os.listdir(DIR_NAME)[1:1001]
celebrity_images = [(DIR_NAME + '/' + photo) for photo in celebrity_photos]

# convert each image to a 64x64 matrix
print("Loading images...")
images = np.array([plt.imread(image) for image in celebrity_images], dtype=np.float64)
print("Images loaded.")

# convert image filenames
celebrity_names = [name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape
#plot_portraits(images, celebrity_names, h, w, n_row=4, n_col=4)

print("Shuffling data...")

# shuffle data to remove bias
paired_data = list(zip(images, celebrity_names))
#random.seed(0)
random.shuffle(paired_data)
images, labels = zip(*paired_data)

images = np.array(images)
labels = np.array(labels)

print("Data shuffled.")

# split data into training and testing data
train_images = images[:TRAIN_SIZE]
train_labels = labels[:TRAIN_SIZE]
test_images = images[TRAIN_SIZE:]
test_labels = labels[TRAIN_SIZE:]

print("Calculating eigenfaces...")

X = train_images.reshape(TRAIN_SIZE, h*w)
P, C, M, Y = pca(X, n_pc=n_components)
eigenfaces = C.reshape((n_components, h, w))
eigenface_titles = ["Eigenface %d" % i for i in range(eigenfaces.shape[0])]

print("Eigenfaces calculated.")
#plot_portraits(eigenfaces, eigenface_titles, h, w, 4, 4)

# plot average of all 800 training images
#plot_portraits(np.array([M]), ["Mean Face"], 64, 64, 1, 1)

##############################################################################

print("Testing model...")

# indices in train_labels of shared names
shared_labels = []

# get all people who are in both training and testing data
for label in test_labels:
    found = np.where(train_labels == label)[0].tolist()
    shared_labels.extend([train_labels[i] for i in found])

# remove all duplicates and get indices for shared
shared_labels = list(set(shared_labels))
shared_indices = [test_labels.tolist().index(label) for label in shared_labels]

print("Shared labels in testing & training: ", shared_labels)

# generate weights as an KxN matrix, where K is number of eigenfaces (50) and N is number of samples (800)
weights = eigenfaces.reshape(50,4096) @ (train_images.reshape(800,4096) - M).T

# one of the names guaranteed to be in both data sets
TEST_FACE = "Bill Clinton"
test_idx = 0
try:    test_idx = test_labels.tolist().index(TEST_FACE)
except: print("Error: chosen face was not found in test data; defaulting to " + test_labels[test_idx])

# allows us to select a custom image for our query
USE_CUSTOM_IMG = False
CUSTOM_IMG = "Bill_Gates_0011.pgm"

# read in custom image or image corresponding to name
img_data = plt.imread(DIR_NAME + '/' + CUSTOM_IMG) if USE_CUSTOM_IMG else test_images[test_idx]

print("Finding best match for " + CUSTOM_IMG if USE_CUSTOM_IMG else TEST_FACE + "...")

# flatten query image into a vector and multiply by weights
query = img_data.reshape(1,-1)
query_weight = eigenfaces.reshape(50,4096) @ (query - M).T

# find the closest match to a training image using weights
euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
best_match = np.argmin(euclidean_distance)

print("Best match: " + train_labels[best_match] + " (cost " + str(euclidean_distance[best_match]) + ")")

# visualize query and best match images
fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
axes[0].imshow(query.reshape(64,64), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(train_images[best_match].reshape(64,64), cmap="gray")
axes[1].set_title("Best match")
#plt.show()

##############################################################################

print("Calculating accuracy of model...")
num_success = 0

# calculate accuracy for all shared labels
for idx in shared_indices:
    # read in image corresponding to test index
    img_data = test_images[idx]

    print("Finding best match for " + test_labels[idx] + "...")

    # flatten query image into a vector and multiply by weights
    query = img_data.reshape(1,-1)
    query_weight = eigenfaces.reshape(50,4096) @ (query - M).T

    # find the closest match to a training image using weights
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)

    print("Best match: " + train_labels[best_match] + " (cost " + str(euclidean_distance[best_match]) + ")")

    if train_labels[best_match] == test_labels[idx]: num_success += 1 

# calculate accuracy of model
accuracy = num_success / len(shared_indices)
print("Accuracy: " + str(accuracy))
f = open("accuracy.txt", 'a')
f.write(str(accuracy) + '\n')
f.close()

print("Done!")