from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import pandas as pd
from numpy.random import choice

def load_movielens(mean_adjust = 'user'):
	'''
		Loads movielens set into a Dataframe. Adjusted rating is user by default
	'''
	path = 'ml-latest-small\\'
	df = pd.read_csv(path + 'ratings.csv')
	movie_xref = pd.read_csv(path + 'movies.csv')
	users = df.groupby('userId').mean()['rating'].reset_index()
	users['user_mean_rating'] = users['rating']
	users['uid'] = users.index
	users.drop('rating',inplace=True,axis=1)
	df = df.merge(users,left_on='userId',right_on='userId')
	movies = df.groupby('movieId').mean()['rating'].reset_index()
	movies['movie_mean_rating'] = movies['rating']
	movies = movies.merge(movie_xref,left_on='movieId',right_on='movieId')
	movies.drop('rating',inplace=True,axis=1)
	movies['mid'] = movies.index
	df = df.merge(movies,left_on='movieId',right_on='movieId')
	if mean_adjust == 'user':
		df['adjusted_rating'] = df['rating'] - df['user_mean_rating']
	elif mean_adjust == 'movie':
		df['adjusted_rating'] = df['rating'] - df['movie_mean_rating']
	return df
	
def load_movielens_sparse(mean_adjust = 'user'):
	'''
		Loads movielens set into a sparse matrix. Adjusted rating is user by default
	'''
	df = load_movielens(mean_adjust)
	users_list = df['uid']
	movies_list = df['mid']
	ratings_list = df['rating']
	df_sparse = csr_matrix((ratings_list,(users_list,movies_list)))
	return df_sparse
	

def train_test_split(sparse_matrix,pct):
	'''Splits a sparse matrix into two sets - a train set and a test set'''
	nz_coord = list(zip(*sparse_matrix.nonzero()))
	nz_coord_array = np.array(nz_coord)
	num_rand = int(np.floor(len(nz_coord)*(pct/100.0)))
	n = len(nz_coord)
	c = choice(n,size=num_rand,replace=False)
	rows = nz_coord_array[c][:,0]
	cols = nz_coord_array[c][:,1]
	vals = np.array(sparse_matrix[rows,cols]).flatten()
	train = sparse_matrix.copy()
	test = lil_matrix(train.shape)
	test[rows,cols] = vals
	test = test.tocsr()
	train[rows,cols] = 0
	train.eliminate_zeros()
	return train,test