'''
Date: 3/16/2019
Author: Devendra Dahal
Description: This is python utility script, which has all of the small function for various
	task needed. This is especially designed for cloud masking task but can be used for other propose 
	wherever the functions are useful. 
USAGE: 

'''
import os, sys, subprocess,csv
from glob import glob
import numpy as np
import pandas as pd
import random as rd
import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array

def GetExtent(gt,cols,rows):
	''' Return list of corner coordinates from a geotransform

		@type gt:   C{tuple/list}
		@param gt: geotransform
		@type cols:   C{int}
		@param cols: number of columns in the dataset
		@type rows:   C{int}
		@param rows: number of rows in the dataset
		@rtype:    C{[float,...,float]}
		@return:   coordinates of each corner
	'''
	ext=[]
	xarr=[0,cols]
	yarr=[0,rows]

	for px in xarr:
		for py in yarr:
			x=gt[0]+(px*gt[1])+(py*gt[2])
			y=gt[3]+(px*gt[4])+(py*gt[5])
			ext.append([x,y])

		yarr.reverse()
	return ext

def GetGeoInfo(SourceDS):
	"""
	Take raster image as input and returns various properties
	Args:
		image file					input map image
		
	Return:
		image size with column and row numbers,
		geotransformation information 
		projection information
		image data type
		count of bands in the image
	"""
	NDV 		= SourceDS.GetRasterBand(1).GetNoDataValue()
	xsize 		= SourceDS.RasterXSize
	ysize 		= SourceDS.RasterYSize
	bands	 	= SourceDS.RasterCount
	GeoT 		= SourceDS.GetGeoTransform()
	proj 		= SourceDS.GetProjection()
	DataType 	= SourceDS.GetRasterBand(1).DataType
	# DataType 	= gdal.GetDataTypeName(DataType)
	return xsize, ysize, GeoT, proj, DataType, bands

def stsCalc(fList,OutFile, sts):
	'''This function calucate stats without considering No Data value '''
	## This Gdal driver should be changed depending on file format
	tifDriver = gdal.GetDriverByName('GTiff')
	tifDriver.Register()
	print ('computing %s of > %s...' % (sts, os.path.basename(OutFile).split('.')[0]))
	
	# ds1 	= gdal.Open(fileName, GA_ReadOnly)
	ds1 	= gdal.Open(fList[0])
	xsize, ysize, gT, prj, dType, bnds = GetGeoInfo(ds1)
	
	dsOut = tifDriver.Create(OutFile, xsize, ysize, 1, gdal.GDT_Byte)
	gdal_array.CopyDatasetInfo(ds1,dsOut)
	# bandOut=dsOut.GetRasterBand(1)
	ds1 = None
	
	x_block_size = 366#block_sizes[0]  
	y_block_size = 366#block_sizes[1]
	# TotSize = xsize*ysize
	for i in range(0, ysize, y_block_size):  
		if i + y_block_size < ysize:  
			rows = y_block_size  
		else:  
			rows = ysize - i  
		for j in range(0, xsize, x_block_size):  
			if j + x_block_size < xsize:  
				cols = x_block_size  
			else:  
				cols = xsize - j
				
			ListofArray = []
			for bn in range(0,len(fList)):
				ds2 	= gdal.Open(fList[bn])
				B =ds2.GetRasterBand(1).ReadAsArray(j, i, cols, rows)
				B = B.astype(float)
				B[(B > 200)] = np.nan
				B[(B < 100)] = np.nan
				# B = np.where(B == 0, np.nan, B)
				ListofArray.append(B)
				
			if sts == 'Mean':
				E = np.nanmean(ListofArray,axis = 0)
			elif sts == 'STD':
				E = np.std(ListofArray,axis = 0)
			elif sts == 'Median':
				E = np.nanmedian(ListofArray,axis = 0)

			# driver = gdal.GetDriverByName("ENVI")
			# E = np.nanmean(ListofArray,axis = 0)
			E = np.where(np.isnan(E),0,E)
			E[(E <= 1.0)] = 255.0
			# E = np.where(E < 10.,255.0,E)
			dsOut.GetRasterBand(1).WriteArray(E,j,i)
			E = None
			B= None
					
	# ds.GetRasterBand(1).WriteArray(NHFD1)
	#Close the datasets
	A = None
	B = None
	E = None
	bandOut = None
	dsOut = None

def MergeRaster(fList,OutFile):
	'''This function calucate stats without considering No Data value '''
	## This Gdal driver should be changed depending on file format
	tifDriver = gdal.GetDriverByName('GTiff')
	tifDriver.Register()
	num = len(fList)
	print ('merging %s rasters to > %s...' % (num, os.path.basename(OutFile).split('.')[0]))
	
	# ds1 	= gdal.Open(fileName, GA_ReadOnly)
	ds1 	= gdal.Open(fList[0])
	xsize, ysize, gT, prj, dType, bnds = GetGeoInfo(ds1)
	
	dsOut = tifDriver.Create(OutFile, xsize, ysize, num, dType)
	gdal_array.CopyDatasetInfo(ds1,dsOut)
	# bandOut=dsOut.GetRasterBand(1)
	ds1 = None

	for bn in range(1,len(fList)+1):
		ds2 	= gdal.Open(fList[bn-1])
		B =ds2.GetRasterBand(1).ReadAsArray()
		B = B.astype(float)
		# A = np.concatenate((A,B),axis = 0)
		# ListofArray.append(B)

		dsOut.GetRasterBand(bn).WriteArray(B)
	B= None
	dsOut = None

def get_xy(r, c):
	'''Get (x, y) raster centre coordinate at row, column'''
	x0, dx, rx, y0, ry, dy = rast_gt    
	return(x0 + c*dx + dx/2.0, y0 + r*dy + dy/2.0)

def stratifed_random(inRaster, outCsv, sampleSize):
	"""
	Save sample class values and pixel coordinates as csv from a random stratified
	sample of classes specified
	Args:
		image file					input map image
		full path name for outFile	CSV file to save the sampled points
		Integer value				Sample size to select from each strata
	Return:
		None
	"""
	# tifDriver = gdal.GetDriverByName('GTiff')
	# tifDriver.Register()
	rast_src = gdal.Open(inRaster)
	global rast_gt
	rast_gt = rast_src.GetGeoTransform()
	
	# Get first raster band
	rast_band = rast_src.GetRasterBand(1)

	# Retrieve as NumPy array to do the serious work
	raster = rast_band.ReadAsArray()
	
	classes = np.sort(np.unique(raster))
	counts = [sampleSize]*classes.size
	
	# Initialize outputs
	# Initialize outputs
	strata = np.array([], dtype=np.int)
	rows = np.array([], dtype=np.int)
	cols = np.array([], dtype=np.int)
	#logger.debug('Performing sampling')

	for c, n in zip(classes, counts):
		#logger.debug('Sampling class {c}'.format(c=c))

		# Find pixels containing class c
		row, col = np.where(raster == c)
		

		# Check for sample size > population size
		if n > col.size:
			n = col.size

		# Randomly sample x / y without replacement
		# NOTE: np.random.choice new to 1.7.0...
		# TODO: check requirement and provide replacement
		samples = np.random.choice(col.size, n, replace=False)

		strata = np.append(strata, np.repeat(c, n))
		rows = np.append(rows, row[samples])
		cols = np.append(cols, col[samples])
		
		## create a pandas dataframe
		df = pd.DataFrame()

		for ind,rs,cs in zip(strata,rows,cols):
			x, y = get_xy(rs, cs)
			df = df.append({'x': x,
							'y': y,
							'cell':ind},ignore_index=True)
							
	df.to_csv(outCsv, sep = ',',index=False)

def calAve(a,b):
	## calculate weighed mean
	x = 0
	y = sum(b)
	for aa,bb in zip(a,b):
		x = x + aa* bb
	z = x/y
	return z
	
def text2shp(in_csv,shpout, epsg):
	"""
	This function convert txt file, if it has coordinates
	to ESRI shapefile using gdal ogr functionality
	Args:
		text file					input csv file with x, y and value columns
		full path name for outFile	filename with .shp extention to save the points
		projection					EPSG code for projection, like 4326 is for WGS 84
	Return:
		None
	"""
	''''''
	
	driver = ogr.GetDriverByName("ESRI Shapefile")
	data_source = driver.CreateDataSource(shpout)
	srs = osr.SpatialReference()
	srs.ImportFromEPSG(epsg)

	layer = data_source.CreateLayer("new", srs, ogr.wkbPoint)

	layer.CreateField(ogr.FieldDefn("x", ogr.OFTReal))
	layer.CreateField(ogr.FieldDefn("y", ogr.OFTReal))
	layer.CreateField(ogr.FieldDefn("vals", ogr.OFTReal))
	
	csvO = open(in_csv,'r')
	csvLs = csvO.readlines()
	csvO.close()

	header = csvLs[0]
	csvLs = csvLs[1:]
	
	for lineID in range(len(csvLs)):
		line = csvLs[lineID].replace('"','')
		line = line.split(',')
		## extracting the train error from current line
		x = float(line[1])
		y = float(line[2])
		z = float(line[3])
		
		feature = ogr.Feature(layer.GetLayerDefn())
		feature.SetField("x", x)
		feature.SetField("y", y)
		feature.SetField("vals", z)

		wkt = "POINT(%f %f)" % (float(x) , float(y))

		point = ogr.CreateGeometryFromWkt(wkt)

		feature.SetGeometry(point)
		layer.CreateFeature(feature)
		feature.Destroy()

	data_source.Destroy()
	
def ModifyCSVCol(inCsv1, NewVal, Year):
	"""
	Modify csv file by adding two columns and assgining values
	Args:
		csv file			input csv file
		integer value			value for new column
		integer value			value for new column
		
	Return:
		None
	"""

	# print ('Adding Assignment:',inCsv)
	in_csv = pd.read_csv(inCsv1, low_memory=False)

	in_csv['cell'] = NewVal
	in_csv.insert(3,'Year', Year)
	
	in_csv.to_csv(inCsv1, sep = ',',index=False)
	
def split_data(in_csv,percent):
	'''This function split input csv with train and test data file
	Train data will contain percent provided as input (provide as fraction e.g 0.8)'''
	
	## reading csv
	csvO = open(in_csv,'r')
	csvLs = csvO.readlines()
	csvO.close()

	csvLs = csvLs[0:]
	csvLen = len(range(len(csvLs)))
	
	## finding a number 
	size = int(round(csvLen*percent))

	splitTrn = rd.sample(csvLs,size)
	splitTst =list(set(csvLs) - set(splitTrn))
	
	##saving file as train data as .data file
	filename, ext = os.path.splitext(in_csv)
	
	FileTrain = filename+'.data'
	outcsv_train = open(FileTrain, "wb")
	for lineID in splitTrn:
		outcsv_train.write(lineID)
	outcsv_train.close()
	
	##saving file as test data as .test file
	FileTest = filename+'.test'
	outcsv_test = open(FileTest, "wb")
	for lineID in splitTst:
		outcsv_test.write(lineID)
	outcsv_test.close()

def extractRaster2Shp(src_filename,shp_filename):
	'''This function extract raster values to a ESRI shapefile
		using gdal ogr modules'''
	src_ds=gdal.Open(src_filename) 
	gt=src_ds.GetGeoTransform()
	rb=src_ds.GetRasterBand(1)

	ds=ogr.Open(shp_filename)
	lyr=ds.GetLayer()
	intval = []
	for feat in lyr:

		geom = feat.GetGeometryRef()
		mx,my=geom.GetX(), geom.GetY()  #coord in map units

		#Convert from map to pixel coordinates.
		#Only works for geotransforms with no rotation.
		px = int((mx - gt[0]) / gt[1]) #x pixel
		py = int((my - gt[3]) / gt[5]) #y pixel

		intval =rb.ReadAsArray(px,py,1,1)
	return intval

def extValueCSV(depV_RasFile,src_filename,in_csv, outFile):
	# src_filename = '/tmp/test.tif'
	
	out_File = open(outFile, "wb")
	src_ds	=gdal.Open(src_filename) 
	devV	=gdal.Open(depV_RasFile) 
	
	xsize, ysize, gt, proj, DataType, bands = GetGeoInfo(src_ds)

	#Convert from map to pixel coordinates.
	#Only works for geotransforms with no rotation.
	csvO = open(in_csv,'r')
	csvLs = csvO.readlines()
	csvO.close()
	
	for lineID in range(1,len(csvLs)):
		line = csvLs[lineID][:-2]
		line = line.split(',')
		mx = float(line[1])
		my = float(line[2])
		px = int((mx - gt[0]) / gt[1]) #x pixel
		py = int((my - gt[3]) / gt[5]) #y pixel

		# loop through the bands
		line = [str(mx),str(my)]
		# extracting independent variables and adding the list
		for i in xrange(1,bands):
			band = src_ds.GetRasterBand(i) # 1-based index
			# read data and add the value to the string
			data = band.ReadAsArray(px, py, 1, 1)
			line.append(str(data[0,0]))
		
		# extracting dependent variable and adding to the list
		bnd = devV.GetRasterBand(1) 
		data1 = bnd.ReadAsArray(px, py, 1, 1)
		line.append(str(data1[0,0]))
		
		lineTempT = ','.join(line)+'\n'
		out_File.write(lineTempT)

def extValue(src_filename,in_csv, outFile, HList,type):
	# src_filename = '/tmp/test.tif'
	
	# out_File = open(outFile, "wb")
	# header = ','.join(HList)+'\n'
	# out_File.write(header)
	
	src_ds=gdal.Open(src_filename) 
	
	xsize, ysize, gt, proj, DataType, bands = GetGeoInfo(src_ds)
	
	with open(outFile, 'w',newline = '') as csvfile:
		out_File = csv.writer(csvfile, doublequote= False,escapechar=',', quoting=csv.QUOTE_NONE)
		# header = ",".join(HList)
		out_File.writerow(HList)
		#Convert from map to pixel coordinates.
		#Only works for geotransforms with no rotation.
		with open(in_csv) as csvRead:
			csvLs = csv.reader(csvRead, delimiter= ',')
			
			for i, line in enumerate(csvLs):
				if i >= 1:
					mx = float(line[1])
					my = float(line[2])
					px = int((mx - gt[0]) / gt[1]) #x pixel
					py = int((my - gt[3]) / gt[5]) #y pixel

					# loop through the bands
					if type.upper() == 'SINGLE':
						line = [str(line[0]),str(mx),str(my),str(line[3])]
					elif type.upper() == 'TREND':
						line = [str(line[0]),str(mx),str(my)]
					elif type.upper() == 'MERGE':
						# line = [str(int(round(mx*100))),str(int(round(my*100))),str(line[2]),str(line[3]),str(line[4])]
						line = [str(line[3]),str(line[0]),str(line[4])]
					# line = [str(mx),str(my)]
					for i in range(1,bands+1):
						band = src_ds.GetRasterBand(i) # 1-based index
						# read data and add the value to the string
						data = band.ReadAsArray(px, py, 1, 1)
						line.append(str(data[0,0]))
					## line = line[1]
					# lineTempT = ",".join(line)
					out_File.writerow(line)
			
def extractRastersValue2CSV(src_rasterfile,in_csv, outFile):
	'''This function extract cell values from all of the band
		of a raster to a csv file reading x, y from another csv file'''
	
	TempVrt = src_rasterfile.replace("csv",".vrt")
	runCom  = 'gdalbuildvrt.exe -separate -q -overwrite -input_file_list %s %s' % (src_rasterfile, TempVrt)
	subprocess.call(runCom, shell=True)
	
	
	out_File = open(outFile, "wb")
	
	csvO = open(in_csv,'r')
	csvLs = csvO.readlines()
	csvO.close()
	#Iterate through each line pair
	header = csvLs[0]
	csvLs = csvLs[1:]
	header = header.split(',')
	for lineID in range(len(csvLs)):
		line = csvLs[lineID] #.replace('"','')
		line = line.split(',')
		x = line[1]
		y = line[2]
		result = subprocess.check_output('gdallocationinfo.exe -valonly -geoloc %s %s %s' %(TempVrt, x, y))
		rr = result.split("\r\n")

		rr = rr[:-2]
		rr.insert(0,y)
		rr.insert(0,x)
		
		lineTempT = ','.join(rr)+'\n'
		out_File.write(lineTempT)
	out_File.close()
	
def mergeCSVs(InputTables, outFile):
	''' This function will fetch .data files from multiple locations and
		merge to make one huge .data file'''
	print ('\nmerging %d the files' % len(InputTables) )
	
	outcsv_File = open(outFile, "wb")

	for inData in InputTables:
		csvFile = open(inData, "rb")
		csvLs = csvFile.readlines()
		## Coping a header from 1st file in the list
		if inData == InputTables[0]:
			header = csvLs[0]
			outcsv_File.write(header)
		## finding site id from the file and creating a list to loop through
		for lineID in range(1,len(csvLs)):
			line = csvLs[lineID]
			outcsv_File.write(line)
	outcsv_File.close()

def RemoveDuplicates(filename):
	'''This function removes all of the duplicate rows from a csv '''
	print ('\nCleaning if there are any duplicate records' )

	df = pd.read_csv(filename, low_memory = False)

	''' Notes: the `subset=None` means that every column is used 
	to determine if two rows are different; to change that specify
	the columns as an array
	the `inplace=True` means that the data structure is changed and
	the duplicate rows are gone  '''
	df.drop_duplicates(subset=None, inplace=True)

	# Write the original file
	df.to_csv(filename, sep = ',',index=False)
	
def ValueReplace(filename):
	'''This function removes all of the duplicate rows from a csv '''
	print ('\nCleaning if there are any duplicate records')

	df = pd.read_csv(filename, low_memory = False)

	''' Notes: the `subset=None` means that every column is used 
	to determine if two rows are different; to change that specify
	the columns as an array
	the `inplace=True` means that the data structure is changed and
	the duplicate rows are gone  '''
	df.replace({'2200': '?'})

	# Write the original file
	df.to_csv(filename, sep = ',',index=False)
	
def columnJoin(inCsv1, inCsv2):
	'''This function copy column form second csv to
		1st csv based on two common columns, x and y'''

	in_csv = pd.read_csv(inCsv1, low_memory=False)
	asgnMas = pd.read_csv(inCsv2, low_memory=False)
	
	OutCsv = pd.merge(in_csv,asgnMas, how = 'left', on = ["x","y"])

	OutCsv.to_csv(inCsv1, sep = ',',index=False)

def RasterSave(CArray,OutFile,cMap,co, ro,bnd):
	dsOut = tifDriver.Create(OutFile, co, ro, bnd, gdal.GDT_Byte)
	CopyDatasetInfo(cMap,dsOut)
	bandOut=dsOut.GetRasterBand(bnd)
	BandWriteArray(bandOut, CArray)
	bandOut=None

def LayerMask(inFile,Mask,outds,ND):
	
	''' Masking input file by mask layer. It will work with only single band file though.'''
	ds1 	= gdal.Open(inFile)
	
	cl, rs, geot, proj, DataType, bnds = GetGeoInfo(ds1)
	inLayer = ds1.GetRasterBand(1)
	
	''' Reading Mask layer in gdal as array for computation.'''
	ds2 	= gdal.Open(Mask)
	inMask = ds2.GetRasterBand(1)
	inM = inMask.ReadAsArray(0,0,cl,rs)
	
	driver = gdal.GetDriverByName("GTiff")
	dst_ds = driver.Create(outds, cl, rs, bnds, DataType)
	dst_ds.SetGeoTransform(geot)
	dst_ds.SetProjection(proj)
	
	for ba in range(bnds):
		
		band = ds1.GetRasterBand(ba+1) # 1-based index
		inL = band.ReadAsArray(0, 0, cl, rs) #.astype(Int)
		# outVar = numpy.zeros((rs, cl),numpy.float)
		
		cfMask = np.where(inM == 1,inL,ND)
		dst_ds.GetRasterBand(ba+1).WriteArray(cfMask)
	inL=None
	inM=None
	
def LayerMask1(inFile,Mask):
	''' Masking input file by mask layer. It will work with only single band file though.'''
	ds1 	= gdal.Open(inFile)
	cl 	= ds1.RasterXSize
	rs 	= ds1.RasterYSize

	''' Values 64, 128, and 192 define good pixels in QA (fmask band) for Landsat8 '''
	inLayer = ds1.GetRasterBand(1)
	
	ds2 	= gdal.Open(Mask)
	inMask = ds2.GetRasterBand(1)
	
	inL = inLayer.ReadAsArray(0,0,cl,rs)
	inM = inMask.ReadAsArray(0,0,cl,rs)
	
	cfMask = np.where(inM == 1,inL,255)
	
	inL=None
	inM=None
	
	outName = inFile.replace('cNDVI','masked_NDVI')
	RasterSave(cfMask,outName,inFile,cl, rs)