# Manifold Sorting
This module can do two things:
1. Sort a 2D array using manifold learning
2. Explore lower dimension (d2) embeddings of a high dimension matrix (d1), where d1> d2 and d2 \in (1,2,3, ...)

This is a module that uses sklearn's manifold learning to automatically explore all methods,
find the best one, and more importantly **SORT** the data array for better visualization if needed.

## Example
To sort a 2D data matrix `data_to_sort`  based on 1D manifold learning using select methods (see example figures) :

    # data_to_sort is a 2D matrix, each row is a vector of ncolumn dimension space: [x1, x2, ..., xn]
    # Manifold learning methods. First 4 are Locally Linear Embedding.

    from manifold_sorting import manifold_sorting as ms
    methods = ['standard', 'modified', 'hessian', 'ltsa', 'Isomap', 'MDS', 'SpectralEmbedding', 'TSNE']
    for method in methods:
        sorted_data = ms.embedding_sort(data_to_sort, method, method, make_plot=True)

    # If Spectral Embedding gives the best result for both rows and columns.
    sorted_data = ms.embedding_sort(data_to_sort, 'SpectralEmbedding', 'SpectralEmbedding')

    # If Spectral embedding on rows and Isomap on columns give the best result:
    sorted_data = ms.embedding_sort(sorted_data, 'SpectralEmbedding', 'Isomap')

    # If only want to sort rows:
    sorted_data = embedding_sort(sorted_data, 'SpectralEmbedding', None)


To explore low dimension embeddings using all 8 methods available from sklearn and find the best performing (e.g., visually appealing) result and method:

    # `dimension` can be 1, 2, 3 or higher (NOTE: higher dimensions are not visualizable)
    # `methods` gives a list of applied methods.
    # This example uses dimnesion = 1, but can be 1, 2, 3 or higher 
    all_trans_data_rows, methods = ms.learn_embedding_all(data_to_embed, dimension=1)

    
