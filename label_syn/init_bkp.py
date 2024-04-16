"""
We gratefully acknowledge Martin Weigert for providing a reference OpenCL
implementation of a nucleus generator upon which this python script is based.
"""

import numpy as np
import os
import nibabel as nib
import argparse

from itertools import product
from perlin_numpy import generate_perlin_noise_3d


def calculate_mahalanobis_distance(center_coords, covariance_matrices, grid_size=128):
    """
    Efficient calculation of Mahalanobis distances using NumPy broadcasting.

    Parameters
    ----------
    center_coords : np.array
        (N_spheres x 3) matrix containing coordinates of sphere centers.
    covariance_matrices : np.array
        (N_spheres x 3 x 3) array of covariance matrices, one for each sphere.
    grid_size : int, optional
        Size along 1 dimension of image. The default is 128.

    Returns
    -------
    distances : np.array
        (N_voxels X N_spheres) distance matrix where N_voxels is grid_size**3.
    """
    # Create an array of all voxel coordinates (grid_size, grid_size, grid_size, 3)
    x, y, z = np.indices((grid_size, grid_size, grid_size))
    voxel_coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Compute Mahalanobis distances using broadcasting and vectorization
    diff = voxel_coords[:, None, :] - center_coords
    distances = np.zeros((voxel_coords.shape[0], center_coords.shape[0]))
    for i in range(center_coords.shape[0]):
        inv_cov = np.linalg.inv(covariance_matrices[i])
        distances[:, i] = np.sqrt(np.einsum('ij,jk,ik->i', diff[:, i, :], inv_cov, diff[:, i, :]))

    return distances
    

def create_random_correlation_matrices(n_matrices, dim):
    """
    Create multiple random correlation matrices of specified dimension with diagonal entries equal to 1
    and all elements not exceeding 1 in absolute value.

    Parameters
    ----------
    n_matrices : int
        Number of correlation matrices to generate.
    dim : int
        Dimension of each correlation matrix (number of variables).

    Returns
    -------
    list of np.array
        List containing n_matrices correlation matrices, each of dim x dim size with diagonal entries equal to 1.
    """
    matrices = []
    for _ in range(n_matrices):
        # Initialize a symmetric matrix with random values
        A = np.random.uniform(0, 1, (dim, dim))
        A = (A + A.T) / 2  # Make symmetric to ensure real eigenvalues
        np.fill_diagonal(A, 1)  # Fill diagonal with 1s for unit variance

        # Use Schur (or spectral) decomposition to project it onto the nearest correlation matrix
        # that is positive semi-definite
        _, s, V = np.linalg.svd(A)  # Singular value decomposition
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A = (A + H) / 2
        A = (A + A.T) / 2  # Ensure the matrix is symmetric

        # Project the eigenvalues to positive by setting all diagonal entries to max(eigenvalue, 0)
        eigval, eigvec = np.linalg.eigh(A)
        eigval[eigval < 0] = 0
        A = np.dot(eigvec, np.dot(np.diag(eigval), eigvec.T))

        # Renormalize to set diagonals to 1
        D_inv = np.sqrt(1 / np.diag(A))
        A = np.dot(np.diag(D_inv), np.dot(A, np.diag(D_inv)))
        np.fill_diagonal(A, 1)

        matrices.append(A)

    return matrices
    


def initial_label_generator(grid_size=128, r_mean=12):
    """
    Parameters
    ----------
    grid_size : int, optional
        Size along 1 dim of image. The default is 128.
    r_mean : int, optional
        Average sphere radius in voxels. The default is 12.

    Returns
    -------
    labelmap : np.array
        (grid_size x grid_size x grid_size) initial labels for synthesis.
    """

    Nz, Ny, Nx = (grid_size,) * 3
    
    x = np.arange(0, grid_size, 2*r_mean)[1:-1]
    Z, Y, X = np.meshgrid(x, x, x, indexing='ij')  # center coordinates
    
    # Randomly translate centers:
    points = np.stack(
        [Z.flatten(), Y.flatten(), X.flatten()]
    ).T # center coordinates flattened
    points = (points).astype(np.float32) 
    points_perturbed = points + .5*r_mean*np.random.uniform(-1,1,points.shape)
    
    # Randomly drop between 0--33% of spheres:
    ind = np.arange(len(points_perturbed))  # index of individual point
    np.random.shuffle(ind)  # randomly shuffle indices
    
    ind_keep = ind[:np.random.randint(2*len(ind)//3,len(ind))]  # drop indices
    points_perturbed_kept = points_perturbed[ind_keep]  # drop spheres
    
    # Randomly scale radii:
    rads = r_mean * np.random.uniform(.6, 1.2, len(points))  # randomly scale
    rads_kept = rads[ind_keep]  # randomly drop radii

    # Compute n_vox x n_spheres matrix:
    cov_matrix = create_random_correlation_matrices(len(ind_keep), 3)
    dist_mtx = calculate_mahalanobis_distance(points_perturbed_kept, cov_matrix)

    # Sample perlin noise:
    noise_sample = generate_perlin_noise_3d((Nz, Ny, Nx), res=(8, 8, 8))

    # Corrupt distance matrix:
    corr_dist_mtx = (
        dist_mtx + 0.9 * r_mean * noise_sample.flatten()[:, np.newaxis]
    )

    # Label assignment:
    labelmap = np.zeros(grid_size**3, dtype=np.uint16)  # initialize
    for j in range(dist_mtx.shape[0]): 
        finder = np.where(corr_dist_mtx[j, :] < rads_kept)[0]
        if len(finder) > 0:
            # in case of match with more than label, assign to closest:
            value = finder[np.argmin(corr_dist_mtx[j, finder])]
            labelmap[j] = value + 1

    labelmap = np.reshape(labelmap, (grid_size, grid_size, grid_size))
    return labelmap