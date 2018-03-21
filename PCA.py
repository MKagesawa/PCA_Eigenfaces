import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

# Loading Data
infile = open('faces.csv', 'r')
img_data = infile.read().strip().split('\n')
img = [map(int, a.strip().split(',')) for a in img_data]
pixels = []
for p in img:
    pixels += p
faces = np.reshape(pixels, (400, 4096))  # Reshape to 400 X 4096 matrix


image_count = 0


# Normalizes vector x
def normalize(U):
    return U / LA.norm(U)  # Linear Algebra matrix norm


# Reshape row vector to 64 X 64 for plotting
first_face = np.reshape(faces[0], (64, 64), order='F')  # column-major order
image_count += 1
plt.figure(image_count)  # create a new figure
plt.title('First_face')  # set title of current axes
plt.imshow(first_face, cmap=plt.cm.gray)  # display image on axes
# plt.show()


# Display a random face from the 400 dataset
random_face = np.reshape(faces[random.randint(0, 399)], (64, 64), order='F')
image_count += 1
plt.figure(image_count)
plt.title('Random Face')
plt.imshow(random_face, cmap=plt.cm.gray)
# plt.show()


# Find the Average face (mean of all features)
mean_face = np.mean(faces, axis=0)
image_count += 1
plt.figure(image_count)
plt.title('Average Face')
plt.imshow(np.reshape(mean_face, (64, 64), order='F'), cmap=plt.cm.gray)
plt.show()


# Subtract the mean face from the face images
subtracted = np.subtract(faces, mean_face)  # (400x4096)

# Calculate the eigenvalues and eigenvectors of the covariance matrix
l = np.matrix(subtracted) * np.matrix(subtracted.transpose())  # mxm matrix L=AA^T (400x400)
eigenvalues, leigenvectors = np.linalg.eig(l)
eigenvalues = np.sort(-eigenvalues)
eigenvectors = leigenvectors[:, (eigenvalues).argsort()]
#print('eigenvalues', -eigenvalues)


# Project the first 16 principle components (eigenfaces)

Z = subtracted.transpose()*eigenvectors
#For loop for 16 eignefaces
for idx, z in enumerate(Z.transpose()):
    a = normalize(z)
    Z[:, idx] = (normalize(z)).transpose()
    if idx < 16:
        plt.subplot(4, 4, idx+1)
        plt.title('Eigenface ' + str(idx))
        plt.imshow(np.reshape(Z[:, idx], (64, 64), order='F'), cmap=plt.cm.gray)
plt.show()


# Reconstruct the face using x number of PCs
def custom_pc(num):
    first_face_reconstruct = Z[:, 0:num]
    omega = np.matrix(first_face_reconstruct.transpose() * np.reshape(subtracted[0, :], (4096, 1)))
    omega_plot = first_face_reconstruct * omega
    reconstruction = np.add(mean_face.reshape(4096, 1), omega_plot)
    global image_count
    image_count += 1
    plt.figure(image_count)
    plt.title('First Face using first ' + str(num) + ' PCs')
    plt.imshow(np.reshape(reconstruction, (64, 64), order='F'), cmap=plt.cm.gray)
    plt.show()

custom_pc(2)


# Reconstruct the first face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs

# 5
custom_pc(5)

# 10
custom_pc(10)

# 25
custom_pc(25)

# 50
custom_pc(50)

# 100
custom_pc(100)

# 200
custom_pc(200)

# 300
custom_pc(300)

# 399
custom_pc(399)


# Plot the percentage of variance attributed to x number of PC inputs

sum_eigenval = np.sum(eigenvalues)
variance = [(e / sum_eigenval)*100 for e in eigenvalues]
cumulative = np.cumsum(variance)
plt.plot(cumulative)
plt.xlabel('Number of PCs')
plt.ylabel('% of variance explained')
plt.show()