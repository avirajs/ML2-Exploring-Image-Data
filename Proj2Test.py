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
eigenfaces = pca.components_.reshape((n_components, h, w))
plot_explained_variance(pca)

# [5 points] Perform non-linear dimensionality reduction of your image data.





# [20 points]  Compare the representation using non-linear dimensions to using linear dimensions.
#The method you choose to compare dimensionality methods should quantitatively explain which method is better at representing the images with fewer components.
#Be aware that mean-squared error may not be a good measurement for kPCA.
#Do you prefer one method over another? Why?
