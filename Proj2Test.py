from sklearn.decomposition import PCA
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob



# Data Preparation (10 points total)
# [5 points] Read in your images as numpy arrays. Resize and recolor images as necessary.
images = []
for filename in glob.glob('./Image_Sample/*.png'):
    print( filename)
    img = mpimg.imread(f'{filename}')
    images.append(rgb2gray(img))

# [1 points] Visualize several images.
imgplot = plt.imshow(images[(np.random.choice(len(images)))])


##each row is in its own right now so we must flatten
h, w = images[0].shape
#each pixel now has a value
print(len(images[0].flatten()))
# [4 points] Linearize the images to create a table of 1-D image features (each row should be one image).
images_features = []
for image in images:
    images_features.append(image.flatten())
print(images_features)









# Data Reduction (60 points total)
# [5 points] Perform linear dimensionality reduction of the images using principal components analysis. Visualize the explained variance of each component. Analyze how many dimensions are required to adequately represent your image data. Explain your analysis and conclusion.
def plot_explained_variance(pca):
    import plotly
    from plotly.graph_objs import Bar, Line
    from plotly.graph_objs import Scatter, Layout
    from plotly.graph_objs.scatter import Marker
    from plotly.graph_objs.layout import XAxis, YAxis
    plotly.offline.init_notebook_mode() # run at the start of every notebook

    explained_var = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(explained_var)

    plotly.offline.iplot({
        "data": [Bar(y=explained_var, name='individual explained variance'),
                 Scatter(y=cum_var_exp, name='cumulative explained variance')
            ],
        "layout": Layout(xaxis=XAxis(title='Principal components'), yaxis=YAxis(title='Explained variance ratio'))
    })

n_components = 6

X= images_features
print(type(X))
pca = PCA(n_components=n_components)
pca.fit(X.copy())
plot_explained_variance(pca)
lin_dim_eig = pca.components_.reshape((n_components, h, w))
def plot_gallery(images , h, w, n_row=3, n_col=6):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.7 * n_col, 2.3 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)

        plt.xticks(())
        plt.yticks(())

plot_gallery(X, h, w)


imgplot = plt.imshow(lin_dim_eig[2])

plot_gallery(lin_dim_eig, h, w)



def reconstruct_image(trans_obj,org_features):
    low_rep = trans_obj.transform(org_features)
    rec_image = trans_obj.inverse_transform(low_rep)
    return low_rep, rec_image

idx_to_reconstruct = 1
X_idx = X[idx_to_reconstruct]
low_dimensional_representation, reconstructed_image = reconstruct_image(pca,X_idx.reshape(1, -1))


plt.subplot(1,2,1)
plt.imshow(X_idx.reshape((h, w)), cmap=plt.cm.gray)
plt.title('Original')
plt.grid()
plt.subplot(1,2,2)
plt.imshow(reconstructed_image.reshape((h, w)), cmap=plt.cm.gray)
plt.title('Reconstructed from Full PCA')
plt.grid()


# [5 points] Perform non-linear dimensionality reduction of your image data.
from sklearn.decomposition import KernelPCA

n_components = 300
print ("Extracting the top %d eigenfaces from %d faces, not calculating inverse transform" % (n_components, len(X)))

kpca = KernelPCA(n_components=n_components, kernel='rbf',
                fit_inverse_transform=True, gamma=12, # very sensitive to the gamma parameter,
                remove_zero_eig=True)
kpca.fit(X.copy())


idx_to_reconstruct = 2
X_idx = X[idx_to_reconstruct]
low_dimensional_representation, reconstructed_image = reconstruct_image(kpca,X_idx.reshape(1, -1))


plt.subplot(1,2,1)
plt.imshow(X_idx.reshape((h, w)), cmap=plt.cm.gray)
plt.title('Original')
plt.grid()
plt.subplot(1,2,2)
plt.imshow(reconstructed_image.reshape((h, w)), cmap=plt.cm.gray)
plt.title('Reconstructed from Kernal PCA')
plt.grid()


np.array(np.mat("2 2 2; 3 4 4")).shape[0]


# [20 points]  Compare the representation using non-linear dimensions to using linear dimensions.
#The method you choose to compare dimensionality methods should quantitatively explain which method is better at representing the images with fewer components.
#Be aware that mean-squared error may not be a good measurement for kPCA.
#Do you prefer one method over another? Why?
