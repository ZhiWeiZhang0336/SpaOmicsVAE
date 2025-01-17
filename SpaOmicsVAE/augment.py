#!/usr/bin/env python3

# @Author: ChangXu
# @E-mail: xuchang0214@163.com
# @Last Modified by:   ChangXu
# @Last Modified time: 2021-04-22 08:42:54 23:22:34
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from sklearn.decomposition import TruncatedSVD



def cal_spatial_weight(
	data,
	spatial_k = 50,
	spatial_type = "BallTree",
	random_seed=42
	):

	np.random.seed(random_seed)

	if spatial_type == "NearestNeighbors":
		nbrs = NearestNeighbors(n_neighbors = spatial_k+1, algorithm ='ball_tree').fit(data)
		_, indices = nbrs.kneighbors(data)
	elif spatial_type == "KDTree":
		tree = KDTree(data, leaf_size=2) 
		_, indices = tree.query(data, k = spatial_k+1)
	elif spatial_type == "BallTree":
		tree = BallTree(data, leaf_size=2)
		_, indices = tree.query(data, k = spatial_k+1)
	indices = indices[:, 1:]
	spatial_weight = np.zeros((data.shape[0], data.shape[0]))
	for i in range(indices.shape[0]):
		ind = indices[i]
		for j in ind:
			spatial_weight[i][j] = 1
	return spatial_weight

def cal_gene_weight(
	data,
	n_components = 50,
	gene_dist_type = "cosine",
	):
	if isinstance(data, csr_matrix):
		data = data.toarray()
	if data.shape[1] > 500:
		pca = PCA(n_components = n_components)
		data = pca.fit_transform(data)
		gene_correlation = 1 - pairwise_distances(data, metric = gene_dist_type)
	else:
		gene_correlation = 1 - pairwise_distances(data, metric = gene_dist_type)
	return gene_correlation

def cal_gene_weight_ATAC(data, n_components=50, gene_dist_type="cosine"):
    # 检查是否为稀疏矩阵
    if isinstance(data, (csr_matrix, csc_matrix)):
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        data = svd.fit_transform(data)
    else:
        if data.shape[1] > 500:
            pca = PCA(n_components=n_components)
            data = pca.fit_transform(data)
    gene_correlation = 1 - pairwise_distances(data, metric=gene_dist_type)
    return gene_correlation


def cal_weight_matrix(
		adata,
		md_dist_type="cosine",
		gb_dist_type="correlation",
		n_components = 50,
		use_morphological = False,
		spatial_k = 30,
		spatial_type = "BallTree",
		verbose = False,
		random_seed=42,
		use_ATAC=False
		):
	if use_morphological:
		if spatial_type == "LinearRegress":
			img_row = adata.obs["imagerow"]
			img_col = adata.obs["imagecol"]
			array_row = adata.obs["array_row"]
			array_col = adata.obs["array_col"]
			rate = 3
			reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
			reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
			physical_distance = pairwise_distances(
									adata.obs[["imagecol", "imagerow"]], 
								  	metric="euclidean")
			unit = math.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
			physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
		else:
			physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type,random_seed=random_seed)
	else:
		physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type,random_seed=random_seed)
	print("Physical distance calculting Done!")
	print("The number of nearest tie neighbors in physical distance is: {}".format(physical_distance.sum()/adata.shape[0]))
	
	if use_ATAC==False:
	########### gene_expression weight
		gene_correlation = cal_gene_weight(data = adata.X.copy(), 
											gene_dist_type = gb_dist_type, 
											n_components = n_components)
	else:
		gene_correlation = cal_gene_weight_ATAC(data = adata.X.copy(), 
											gene_dist_type = gb_dist_type, 
											n_components = n_components)
	# gene_correlation[gene_correlation < 0 ] = 0
	print("Gene correlation calculting Done!")
 
	if verbose:
		adata.obsm["gene_correlation"] = gene_correlation
		adata.obsm["physical_distance"] = physical_distance


	adata.obsm["weights_matrix_all"] = (gene_correlation
												* physical_distance)
	print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")
 
	return adata

def find_adjacent_spot(
    adata,
    use_data="raw",
    neighbour_k=4,
    verbose=False,
):
    """
    Find adjacent spots and calculate gene matrix based on spatial weights.

    Parameters:
        adata: AnnData object
            Input data object.
        use_data: str
            Data to use (default: "raw").
        neighbour_k: int
            Number of neighbors to consider.
        verbose: bool
            Whether to output detailed information.

    Returns:
        adata: AnnData object
            Updated object with adjacent data.
    """
    if use_data == "raw":
        # Process raw data
        if isinstance(adata.X, csr_matrix):
            gene_matrix = adata.X.toarray()  # CSR -> Dense
        elif isinstance(adata.X, csc_matrix):
            gene_matrix = adata.X.tocsr().toarray()  # CSC -> CSR -> Dense
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X  # NumPy array
        elif isinstance(adata.X, pd.DataFrame):
            gene_matrix = adata.X.values  # DataFrame -> NumPy array
        else:
            raise ValueError(f"Unsupported data type for adata.X: {type(adata.X)}")
    else:
        # Use data from adata.obsm
        gene_matrix = adata.obsm[use_data]

    weights_list = []
    final_coordinates = []

    # Iterate through each spot to find neighbors
    with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
        for i in range(adata.shape[0]):
            # Get top-K neighbors
            current_spot = adata.obsm['weights_matrix_all'][i].argsort()[-neighbour_k:][:neighbour_k - 1]
            spot_weight = adata.obsm['weights_matrix_all'][i][current_spot]
            spot_matrix = gene_matrix[current_spot]
            if spot_weight.sum() > 0:
                spot_weight_scaled = (spot_weight / spot_weight.sum())
                weights_list.append(spot_weight_scaled)
                spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_matrix)
                spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
            else:
                spot_matrix_final = np.zeros(gene_matrix.shape[1])
                weights_list.append(np.zeros(len(current_spot)))
            final_coordinates.append(spot_matrix_final)
            pbar.update(1)

    # Save results in adata.obsm
    adata.obsm['adjacent_data'] = np.array(final_coordinates)
    if verbose:
        adata.obsm['adjacent_weight'] = np.array(weights_list)
    return adata


def augment_gene_data(
    adata,
    adjacent_weight=0.2,
):
    """
    Augment gene expression data by adding weighted adjacent data.

    Parameters:
        adata: AnnData object
            Input data object.
        adjacent_weight: float
            Weight for adjacent data contribution.

    Returns:
        adata: AnnData object
            Updated AnnData object with augmented gene data.
    """
    if isinstance(adata.X, csr_matrix):
        # 如果是稀疏矩阵，保持稀疏计算
        augement_gene_matrix = adata.X + csr_matrix(adjacent_weight * adata.obsm["adjacent_data"])
    elif isinstance(adata.X, csc_matrix):
        # 如果是 CSC 稀疏矩阵，先转换为 CSR
        augement_gene_matrix = adata.X.tocsr() + csr_matrix(adjacent_weight * adata.obsm["adjacent_data"])
    else:
        # 如果是密集矩阵，直接进行加法
        augement_gene_matrix = np.array(adata.X) + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
    
    # 将增强后的数据存储回 AnnData 对象
    adata.obsm["augment_gene_data"] = augement_gene_matrix
    return adata

def augment_adata(
	adata,
	md_dist_type="cosine",
	gb_dist_type="correlation",
	n_components = 50,
	use_morphological = True,
	use_data = "raw",
	neighbour_k = 4,
	adjacent_weight = 0.2,
	spatial_k = 30,
	spatial_type = "KDTree",
 	random_seed=42,
	use_ATAC=False
	):
	adata = cal_weight_matrix(
				adata,
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				use_morphological = use_morphological,
				spatial_k = spatial_k,
				spatial_type = spatial_type,
    			random_seed=random_seed,
				use_ATAC=use_ATAC
				)
	adata = find_adjacent_spot(adata,
				use_data = use_data,
				neighbour_k = neighbour_k)
	adata = augment_gene_data(adata,
				adjacent_weight = adjacent_weight)
	return adata