# -*- coding: utf-8 -*-
'''This is a module that uses sklearn's manifold learning to automatically explore all methods,
find the best one, and more importantly *sort* the data array if needed.

Example
---------
To sort a data matrix `data_to_sort`  based on 1D manifold learning using select methods:
            # If Spectral embedding on gives the best result for both rows and columns.
            # data_to_sort is a 2D matrix, each row is a vector of ncolumn dimension space: [x1, x2, x3..., xn]
            sorted_data = embedding_sort(data_to_sort, 'SpectralEmbedding', 'SpectralEmbedding')

            # If Spectral embedding on rows and Isomap on columns give the best result:
            sorted_data = embedding_sort(sorted_data, 'SpectralEmbedding', 'Isomap')

            # If only want to sort rows:
            sorted_data = embedding_sort(sorted_data, 'SpectralEmbedding', None)


To learn 1-D embedding of all rows in a matrix and explore the results on your own:
            all_trans_data_rows, methods = learn_embedding_all(data_to_embed, dimension=1)


@author: Ming Zhao @ NYT
Created on Mon Jun 20 11:59:31 2016
'''
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D



def learn_embedding(data_to_embed, method, dimension=1, n_neighbors = 10):
    '''Learn embedding for a given method. This is based on sklearn's manifold examples.

    Parameters
    -------------
    data_to_embed : nd.array or pd.DataFrame
                    A matrix to be used for manifold learning. Each row is a vector in the n_column high dimension space,
                    i.e., rows are instances and columns are features or dimensions.
    method : str
            Any of the following methods: 'standard',  'modified', 'hessian', 'ltsa', 'Isomap', 'MDS', 'SpectralEmbedding', 'TSNE'
            Check http://scikit-learn.org/stable/modules/manifold.html for details of all methods.
            *NOTE*: if `learn_embedding_all` is called, then `method` can be retrieved from the returned variables. See below.
    dimension : int
            The final dimension to be learnt and projected to.
    n_neighbors : int, default=10
            Number of points for finding nearest neighbors. Increase to a larger number if see error message that it is too small.

    Return
    -------
    trans_data : np.array or pd.DataFrames
            A matrix or DataFrame transformed/projected into a smaller manifold of given dimension.
            The sequence of instances, i.e., rows in the input data is kept.
    '''
    if method in ['standard',  'modified', 'hessian', 'ltsa']:
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors, dimension, method=method, eigen_solver='dense').fit_transform(data_to_embed).T
    elif method== 'Isomap':
        trans_data = manifold.Isomap(n_neighbors, n_components=dimension).fit_transform(data_to_embed).T
    elif method== 'MDS':
        # Perform Multi-dimensional scaling.
        mds = manifold.MDS(dimension, max_iter=200, n_init=1)
        trans_data = mds.fit_transform(data_to_embed).T
    elif method=='SpectralEmbedding':
        # Perform Spectral Embedding.
        se = manifold.SpectralEmbedding(n_components=dimension, n_neighbors=n_neighbors, affinity='nearest_neighbors')
        trans_data = se.fit_transform(data_to_embed).T
    elif method=='TSNE':
        # Perform t-distributed stochastic neighbor embedding.
        tsne = manifold.TSNE(n_components=dimension, init='pca', random_state=0)
        trans_data = tsne.fit_transform(data_to_embed).T

    return trans_data



def learn_embedding_all(data_to_embed, dimension=1, n_neighbors=10):
    '''To learn embedding using all 8 methods available from sklearn. This is used for exploration of all methods to find
    the best performing (e.g., visually appealing) result and method.

    Parameters
    -------------
    data_to_embed : nd.array or pd.DataFrame
                    A matrix to be used for manifold learning. Each row is a vector in the n_column high dimension space,
                    i.e., rows are instances and columns are features or dimensions.
    dimension : int
            The final dimension to be learnt and projected to.
    n_neighbors : int, default=10
            Number of points for finding nearest neighbors. Increase to a larger number if see error message that it is too small.

    Return
    -------
    trans_data : np.array or pd.DataFrames
            A matrix or DataFrame transformed/projected into a smaller manifold of given dimension.
            The sequence of instances, i.e., rows in the input data is kept.
    methods : list
             List of methods applied to data.

    Example
    ----------
    To learn 1-D embedding of all rows in a matrix:
            all_trans_data_rows, methods = learn_embedding_all(data_to_embed, dimension=1)

    '''
    if dimension >3:
        print('dimension must be <=3 for visualization!! Quit.')                
        return None                    
    n_samples = data_to_embed.shape[0]
    all_trans_data = []

    # Manifold learning methods. First 4 are Locally Linear Embedding.
    methods = ['standard',  'modified', 'hessian', 'ltsa', 'Isomap', 'MDS', 'SpectralEmbedding', 'TSNE']
    labels = ['LLE',  'Modified LLE', 'Hessian', 'LTSA', 'Isomap', 'MDS', 'SpectralEmbedding', 'TSNE']

    for method in methods:
        trans_data = learn_embedding(data_to_embed, method, dimension, n_neighbors=n_neighbors)
        all_trans_data.append(trans_data)


    # Plotting
    Axes3D
    fig = plt.figure(figsize=(17, 10))
    plt.suptitle("Manifold Learning with {0} points, {1} neighbors".format(n_samples, n_neighbors), fontsize=14)
    colors = cm.rainbow(np.linspace(0,1, n_samples))

    for i, method in enumerate(methods):
        trans_data = all_trans_data[i]
        if dimension==3:
            elev = 30
            z_angle=60
            ax = fig.add_subplot(241 + i, projection='3d')
            ax.scatter(trans_data[0], trans_data[1], trans_data[2],  c=colors, cmap=plt.cm.rainbow)
            ax.view_init(elev, z_angle)
        elif dimension==2:
            ax = fig.add_subplot(241 + i)
            ax.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
        elif dimension==1:
            ax = fig.add_subplot(241 + i)
            ax.scatter(range(len(trans_data[0])), trans_data[0], c=colors, cmap=plt.cm.rainbow)
        plt.title("{0}".format(labels[i]))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
    plt.show()

    return all_trans_data, methods



def embedding_sort(data_to_embed, method_for_row, method_for_col, n_neighbors = 10, make_plot=True):
    '''Sort 2-D data by learning 1d embedding using all methods in sklearn. Rows are sorted first, then columns.
    This function can be called after first using `learn_embedding_all()` to explore the best methods.

    Parameters
    -------------
    data_to_embed : nd.array or pd.DataFrame
                    A matrix to be used for manifold learning. Each row is a vector in the n_column high dimension space,
                    i.e., rows are instances and columns are features or dimensions.
    method_for_row : str or None
                    Any of the following methods: 'standard',  'modified', 'hessian', 'ltsa', 'Isomap', 'MDS', 'SpectralEmbedding', 'TSNE'
                    Check http://scikit-learn.org/stable/modules/manifold.html for details of all methods.
                    A `methods` list can also be returned by calling `learn_embedding_all` first.
                    If == None, then no sorting will be apllied to rows.


    method_for_col : str or None
                    Similar to above but for columns, if use different methods for rows and columns.
                    This is not necessary but can help for some data.
                    If == None, then no sorting will be apllied to columns.

    Return
    --------
    data_to_embed_all_sorted : np.array or pd.DataFrame
                    The sorted 2D array.

    Example
    --------
    To use this for a 2D matrix `data_to_sort` using different methods:
            # If Spectral embedding gives the best result:
            sorted_data = embedding_sort(data_to_sort, 'SpectralEmbedding', 'SpectralEmbedding')
            # If Spectral embedding on rows and Isomap on columns give the best result:
            sorted_data = embedding_sort(sorted_data, 'SpectralEmbedding', 'Isomap')
    '''
    shape = data_to_embed.shape

    # Step 1: learn embedding of rows and sort rows
    if method_for_row == None:
        data_to_embed_row_sorted = data_to_embed
    else:
        trans_data_row = learn_embedding(data_to_embed, method_for_row, n_neighbors= n_neighbors)
        ind_row = np.argsort(trans_data_row[0])[::-1]
        if isinstance(data_to_embed, np.ndarray):
            data_to_embed_row_sorted = data_to_embed[ind_row, :]
        elif isinstance(data_to_embed, pd.DataFrame):
            data_to_embed_row_sorted = data_to_embed.iloc[ind_row, :]

    # Step 2: learn embedding of columns and sort columns
    if method_for_col==None:
        data_to_embed_all_sorted = data_to_embed_row_sorted
    else:
        trans_data_col = learn_embedding(data_to_embed_row_sorted.T, method_for_col, n_neighbors=n_neighbors)
        ind_col = np.argsort(trans_data_col[0])[::-1]
        if isinstance(data_to_embed_row_sorted, np.ndarray):
            data_to_embed_all_sorted = data_to_embed_row_sorted[:, ind_col]
        elif isinstance(data_to_embed_row_sorted, pd.DataFrame):
            data_to_embed_all_sorted = data_to_embed_row_sorted.iloc[:, ind_col]

    if make_plot==True:
        # Plotting
        fig, axes = plt.subplots(2,2, figsize=(12, 10))
        plt.suptitle("{0}+{1} Manifold Learning with {2} points, {3} neighbors".format(method_for_row,
                                                                                       method_for_col,
                                                                                       shape[0], n_neighbors), fontsize=14)
        row_colors = cm.rainbow(np.linspace(0,1, shape[0]))
        col_colors = cm.rainbow(np.linspace(0,1, shape[1]))

        axes[0,0].scatter(range(len(trans_data_row[0])), trans_data_row[0], c=row_colors, cmap=plt.cm.rainbow)
        axes[0,0].set_title('1d Embedding for rows')
        axes[0,1].scatter(range(len(trans_data_col[0])), trans_data_col[0], c=col_colors, cmap=plt.cm.rainbow)
        axes[0,1].set_title('1d Embedding for columns')
        axes[1,0].imshow(data_to_embed_row_sorted, cmap='jet')
        axes[1,0].grid(False)
        axes[1,0].set_title('Sorted heatmap for sorted rows')
        axes[1,1].imshow(data_to_embed_all_sorted, cmap='jet')
        axes[1,1].grid(False)
        axes[1,1].set_title('Sorted heatmap for both rows and columns sorted')
        plt.show()

    return data_to_embed_all_sorted
