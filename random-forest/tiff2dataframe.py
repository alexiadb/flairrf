#!/bin/python3
#This file transforms a tiff image to a dataframe

import os
import logging
import glob 
import pandas as pd
from osgeo import gdal_array
import numpy as np

def set_environment(global_vars: dict) -> int:
	''' Sets all the variables for the environments into a params variables

	params:
	-------
		global_vars : 	dict : all the variables will be stored in this dictionnary

	returns:
		0 			: standard linux return's value without an error

	'''

	global_vars['inputdir']='/inputdir'
	global_vars['outputdir']='/outputdir'
	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
	global_vars['logger'] = logging.getLogger(__name__)
	global_vars['cal_ndvi'] = True

	
	return 0



desc = """
	Starting Transformation function from Tiff to DataFrame
	inputs : /inputdir : full paths of the input tiff files
	 	/outputdir : directory of the outputs files   
"""

def output_file(global_var: dict, input_file: str) -> str:
	'''function to write to output filename of the pandas dataframe from an input file
	'''

	ifilename = os.path.basename(input_file).split('.')[0]
	outputdir = global_var['outputdir']

	ofilename =  os.path.join(outputdir,"{0}.csv".format(ifilename))

	return ofilename

def tiff2df_3D(global_var : dict, input_file: str) -> pd.DataFrame : 
	'''Reads a tiff file full path, and generates a pandasDataFrame from it
		The functions uses the libraries gdal (osgeo)
		https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
	'''

	logger = global_var['logger']
	logger.info('reading input file {0}'.format(input_file))

	np_array = gdal_array.LoadFile(input_file)

	z_dim,x_dim,y_dim = np.shape(np_array)

	values = np.transpose(np_array.reshape(z_dim,-1)) / 255.
	i_rows = np.array( [[j for i in range(x_dim)] for j in range(y_dim)]).flatten()
	j_cols = np.array(x_dim * [j for j in range(y_dim)])

	df = pd.DataFrame(data = values, columns=['R','G','B', 'N', 'H'])

	df['i_rows'] = i_rows
	df['j_cols'] = j_cols

	return df

def tiff2df_1D(global_var : dict, input_file: str) -> pd.DataFrame : 
	'''Reads a tiff file full path, and generates a pandasDataFrame from it
		The functions uses the libraries gdal (osgeo)
		https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
		of 2 dimensional array (for class)
	'''

	logger = global_var['logger']
	logger.info('reading input file {0}'.format(input_file))

	np_array = gdal_array.LoadFile(input_file)

	x_dim,y_dim = np.shape(np_array)
	
	values = np_array.reshape(-1) 
	i_rows = np.array( [[j for i in range(x_dim)] for j in range(y_dim)]).flatten()
	j_cols = np.array(x_dim * [j for j in range(y_dim)])

	df = pd.DataFrame(data = values, columns=['C'])

	df['i_rows'] = i_rows
	df['j_cols'] = j_cols

	return df

def save(global_var : dict, idf : pd.DataFrame, input_file : str) -> int : 
	'''save dataframe to a file in the output directory
	'''
	ofilename = output_file(global_var, input_file) + '.gzip'

	logger.info('saving to file: {0}'.format(ofilename))

	idf.to_csv(path_or_buf=ofilename, 
		index=False, 
		compression={'method': 'gzip', 'compresslevel': 6}) 

	return 0

def ndvi(global_var: dict, idf: pd.DataFrame) -> pd.DataFrame: 
	'''function to calculate ndvi index from a dataframe
	'''
	idf['ndvi'] = (idf['R']-idf['G']) / (idf['R']+idf['G'])

	return idf

def do_msk(global_var : dict) -> int:
	'''Performs mask only it is assumed a 1D array in input
	'''
	#read file structure

	inputdir = global_var['inputdir']
	input_files = glob.glob(os.path.join(inputdir,'**','MSK_*.tif'), recursive=True)

	for input_file in input_files:
		df=tiff2df_1D(global_var, input_file)		
		save(global_var, df,input_file)

	return 0

def do_train(global_var : dict) -> int:
	'''Performs train only 
	'''
	logger = global_var['logger']
	inputdir = global_var['inputdir']
	cal_ndvi = global_var['cal_ndvi']


	#read file structure

	input_files = glob.glob(os.path.join(inputdir,'**','IMG_*.tif'), recursive=True)

	for input_file in input_files:
		df=tiff2df_3D(global_var, input_file)

		if cal_ndvi:
			df = ndvi(global_var, df)	
		
		save(global_var, df,input_file)

	return 0

if __name__ == '__main__':
	
	global_var = {}
	set_environment(global_var)
	logger = global_var['logger']
	logger.info(desc)

	logger.info('global_var {0}'.format(global_var))

	logger.info('transforming training ... ')
	do_train(global_var)

	logger.info('transforming mask ...')
	do_msk(global_var)


	




