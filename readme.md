# Manifold Sorting
Sort an array using manifold learning

This is a module that uses sklearn's manifold learning to automatically explore all methods,
find the best one, and more importantly **SORT** the data array for better visualization if needed.

## Example
To sort a data matrix `data_to_sort`  based on 1D manifold learning using select methods (see example figures) :

    # data_to_sort is a 2D matrix, each row is a vector of ncolumn dimension space: [x1, x2, ..., xn]
    # Manifold learning methods. First 4 are Locally Linear Embedding.

    import manifold_sorting as ms
    methods = ['standard', 'modified', 'hessian', 'ltsa', 'Isomap', 'MDS', 'SpectralEmbedding', 'TSNE']
    for method in methods:
        sorted_data = ms.embedding_sort(data_to_sort, method, method)

    # If Spectral Embedding gives the best result for both rows and columns.
    sorted_data = ms.embedding_sort(data_to_sort, 'SpectralEmbedding', 'SpectralEmbedding')

    # If Spectral embedding on rows and Isomap on columns give the best result:
    sorted_data = ms.embedding_sort(sorted_data, 'SpectralEmbedding', 'Isomap')


To learn 1-D embedding of all rows in a matrix and explore the results later on your own:

    # `dimension` can be 1, 2, 3 or higher (NOTE: higher dimensions are not visualizable)
    # `methods` gives a list of applied methods.
    all_trans_data_rows, methods = ms.learn_embedding_all(data_to_embed, dimension=1)
