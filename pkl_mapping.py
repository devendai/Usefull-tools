'''
Date: 9/4/2020
Author: Devendra Dahal
Update:
Description: read pickle file, text file with list of raster file with full path and 
	make predicted raster file. This code requires utilities.py 
USAGE: 
'''

import os,sys,traceback, datetime,time, itertools,random
from glob import glob
import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array
from osgeo.gdalconst import GA_ReadOnly
from osgeo import osr
from sklearn.externals import joblib
import pickle
import utilities as util

t1 = datetime.datetime.now()

def func(z):
	
	rows		= z[0]
	cols		= z[1]
	j			= z[2]
	i			= z[3]
	TotBands	= z[4]
	GetOneImg	= z[5]
	Raster_dict	= z[6]
	clf 		= z[7]
	dst_df		= z[8]
	
	# print(clf.estimators_[0].feature_importances_)
	# Need to set up however many rasters you need...
	img = np.zeros((rows, cols, TotBands), dtype=np.int16)
	
	count = - 1
	Band1 = GetOneImg[0]
	Img_ds = gdal.Open(Band1, gdal.GA_ReadOnly)

	# Add to stacked image
	dArry = Img_ds.GetRasterBand(1).ReadAsArray(j, i, cols, rows)

	for item in Raster_dict.items():
		count = count + 1

		key = item[0]
		InImg = item[1][-1]
		Band = item[1][1]
		
		# print(InImg)
		# Open Image
		Img_ds = gdal.Open(InImg, gdal.GA_ReadOnly)
		# Add to stacked image
		img[:, :, count] = Img_ds.GetRasterBand(Band).ReadAsArray(j, i, cols, rows)

	# Take our full image, 
	new_shape = (img.shape[0] * img.shape[1], img.shape[2])
	# print(img)
	img_as_array = img[:, :, :].reshape(new_shape)
	# Now predict for each pixel
	# print(img_as_array)
	class_prediction = clf.predict(img_as_array)
	
	# Reshape our classification map
	class_prediction = class_prediction.reshape(img[:, :, 0].shape)
	
	# Write array
	dst_df.GetRasterBand(1).WriteArray(class_prediction,j,i)
	
	class_prediction = None 
	img_as_array = None
	img = None
	
	return class_prediction

def runMapping(OutTif,Raster_Dir_List,InVars,clf):
	
	Raster_info_dict = {}
	GetOneImg = []
	print('.....List of input variables.......')
	for VarOut, r in zip(InVars,Raster_Dir_List):
		ds1 	= gdal.Open(r)
		xsize, ysize, GeoT, Projection, DataType, BandNum = util.GetGeoInfo(ds1)

		print('...'+VarOut+'...')
		
		key = VarOut
		Raster_info_dict[key] = ( DataType,BandNum, xsize, ysize, GeoT, Projection, r )

		# Only append 1st img
		if GetOneImg == []:
			GetOneImg.append(r)
		else:
			pass
	TotBands = len(Raster_info_dict)
	### Open one image for info
	Img1_ds = gdal.Open(GetOneImg[0])
	
	#Read in Block sizes
	imgG = Img1_ds.GetRasterBand(1)
	block_sizes = imgG.GetBlockSize()  
	x_block_size = 500#block_sizes[0]  
	y_block_size = 500#block_sizes[1]

	# Set up output Files
	dst_ds = create_outRaster(OutTif, xsize, ysize,Img1_ds, DataType)
	
	# p = Pool(util.cpu_avail())
	# p = Pool(1)
	# margs = []
	for i in tqdm(range(0, ysize, y_block_size)):  
		if i + y_block_size < ysize:  
			rows = y_block_size  
		else:  
			rows = ysize - i  
		for j in range(0, xsize, x_block_size):  
			if j + x_block_size < xsize:  
				cols = x_block_size  
			else:  
				cols = xsize - j
			
			m_item = (rows,cols,j,i,TotBands,GetOneImg,Raster_info_dict,clf,dst_ds)
			
			pool.apply_async(worker,func(m_item))
			
	dst_ds = None
	Img1_ds = None

def runPro(Dir,OutRtr):
	try:
		pklFile 	= glob(Dir + os.sep + '*.pkl')[0]
		rlistFile 	= glob(Dir + os.sep + '*.rlist')[0]
		BaseOutName = Dir + os.sep + OutRtr	
		
		# Apply model to new dataset...
		clf = joblib.load(pklFile)
		
		OutTif =  BaseName + '_pred.tif'
		if not os.path.exists(OutTif):

			print('\nCreating 2 raster files at {} !\n'.format(Dir))
		
		# Get List of Raster Available
		# File list way
		#Raster_Dir_List = GetRasterList(RasDir)
		# Create list of rasters from
		Raster_Dir_List = []
		InVars = []
		
		csv1 = open(rlistFile,'r')
		csvLs = csv1.readlines()
		csv1.close()
		
		for line in csvLs:
			if 'ignore' in line:
				pass
			else:
				# Sline = line.split('\t')
				txtlist = (line.split(' '))
				Image_loc = txtlist[1].replace("\n","") #.split('.')[0]+'.tif'
				if Image_loc not in Raster_Dir_List:
					InVars.append(txtlist[0])
					if os.path.exists(Image_loc):
						Raster_Dir_List.append(Image_loc)
					else:
						print(Image_loc +' does not exist..Bailing!!!')
						sys.exit()
				else:
					pass

		runMapping(OutTif,Raster_Dir_List,InVars,clf)
		
	except:
		print ("Processed halted on the way.")
		print (traceback.format_exc())

'''
Dir 	= sys.argv[1] ##  example: Y:\CFLUXFS2\ddahal\AnnHerb_extend\temp\hybrid2_itr3
OutRstr 	= sys.argv[2] ## AnnHerb_AIM2016_traning_itr3_xgboost_test.tif
runPro(Dir,OutRstr)
'''

t2 = datetime.datetime.now()
print (t2.strftime("%Y-%m-%d %H:%M:%S"))
tt = t2 - t1
print ("\nProcessing time: " + str(tt))