#!/bin/python
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
	add the classes ground true label to the  csv dataframe  
	inputs : /inputdir : full paths of the msk and img csv files
	 	/outputdir : directory of the outputs csv files   
"""

def do_combine(globar_var: dict) -> int:
	'''creates the training set from an img csv and a mask csv
	'''

	inputdir = global_var['inputdir']
	outputdir = global_var['outputdir']
	logger = global_var['logger']

	msk_files = glob.glob(os.path.join(inputdir,'**','MSK_*.csv.gzip'), recursive=True)

	for msk_file in msk_files: 
		img_file = msk_file.replace('MSK','IMG')

		if os.path.exists(img_file):
			logger.info('combining {0}'.format(msk_file))

			df_train = pd.read_csv(img_file, compression={'method': 'gzip', 'compresslevel': 6})
			df_msk = pd.read_csv(msk_file, compression={'method': 'gzip', 'compresslevel': 6})
			df_train['C'] = df_msk['C']
			ofilename=os.path.join(outputdir,os.path.basename(img_file))

			df_train.to_csv(path_or_buf=ofilename, 
					index=False, 
					compression={'method': 'gzip', 'compresslevel': 6})
		else:
			logger.warning('corresponding img of {0} not found'.format(msk_file))

	return 0


if __name__ == '__main__':
	
	global_var = {}
	set_environment(global_var)
	logger = global_var['logger']
	logger.info(desc)

	logger.info('global_var {0}'.format(global_var))

	logger.info('combining training ... ')
	do_combine(global_var)


	




