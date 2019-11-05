'''
Description: This script is to fix spike in C6 of eMODIS NDVI collection taking median of +_ 2 from current week
	and comparing with the real value
Author: Devendra Dahal 
	I (Devendra Dahal) wrote the script but the algorithm was developed by Bruce Wylie and myself. And he was the
	reason this script/algorithm was developed, therefore, the credit should to him as well.
	(Suggested credit: Devendra Dahal and Bruce Wylie, USGS EROS Center)
Date:	4/3/2017
Last Updated: 11/5/2019: conversion to python 3, pervious version was python 2.7


'''
import os, sys, traceback, csv, time, datetime, subprocess, glob
from optparse import OptionParser 
import utility_eM as util
try:
	from osgeo import gdal
	from osgeo.gdalconst import *
	from osgeo.gdalnumeric import *
except ImportError:
	import gdal	

t1 = datetime.datetime.now()
print (t1.strftime("%Y-%m-%d %H:%M:%S"))

tifDriver = gdal.GetDriverByName('GTiff')
tifDriver.Register()
	
def tMax(fileName,li,OutFile):
	ds1 	= gdal.Open(fileName)
	
	

	xsize, ysize, GeoT, proj, Dtype, bands = util.GetGeoInfo(ds1)
	
	dsOut = tifDriver.Create(OutFile, xsize, ysize, 1, Dtype)
	CopyDatasetInfo(ds1,dsOut)
	
	x_block_size = 5000#block_sizes[0]  
	y_block_size = 5000#block_sizes[1]
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
				
			#Read the data into numpy arrays
			A = ds1.GetRasterBand(li[0]).ReadAsArray(j, i, cols, rows)
			B = ds1.GetRasterBand(li[1]).ReadAsArray(j, i, cols, rows)
			C = ds1.GetRasterBand(li[4]).ReadAsArray(j, i, cols, rows)
			D = ds1.GetRasterBand(li[5]).ReadAsArray(j, i, cols, rows)
			
			E = numpy.amax([A,B,C,D],0)
		
			dsOut.GetRasterBand(1).WriteArray(E,j,i)
	# ds.GetRasterBand(1).WriteArray(NHFD1)
	#Close the datasets
	A = None
	B = None
	D = None
	C = None
	E = None
	bandOut = None
	dsOut = None

def dspike_edge(fileName,Max,Num1,Num2,OutFile):
	ds1 	= gdal.Open(fileName, GA_ReadOnly)
	
	xsize, ysize, GeoT, proj, Dtype, bands = util.GetGeoInfo(ds1)
	
	ds2 	= gdal.Open(Max, GA_ReadOnly)
	
	dsOut = tifDriver.Create(OutFile, xsize, ysize, 1, Dtype)
	CopyDatasetInfo(ds1,dsOut)
	
	x_block_size = 2000#block_sizes[0]  
	y_block_size = 2000#block_sizes[1]
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
				
			A 	= ds1.GetRasterBand(Num1).ReadAsArray(j, i, cols, rows)
			B 	= ds1.GetRasterBand(Num2).ReadAsArray(j, i, cols, rows)
			C	= ds2.GetRasterBand(1).ReadAsArray(j, i, cols, rows)
			
			AC = numpy.subtract(A*1.0,C)
			Bn = numpy.multiply(B,1.1)
			
			# print 'A.shape is %s' %str(A.shape)
			E = numpy.zeros((rows, cols), dtype=numpy.uint8)
			E[numpy.logical_and((A >= Bn),(AC <= 20))] = A[numpy.logical_and((A >= Bn),(AC <= 20))]
			E[(A < B*1.1)] = A[(A < B*1.1)]
			
			dsOut.GetRasterBand(1).WriteArray(E,j,i)
	
	AC = None
	Bn = None
	band1 = None
	bandN1 = None
	bandMax = None
	ds1 = None
	ds2 = None
	bandOut = None
	dsOut = None

def dspike(fileName,Max,Num1,Num2,Num3,OutFile):
	ds1 	= gdal.Open(fileName, GA_ReadOnly)
	
	xsize, ysize, GeoT, proj, Dtype, bands = util.GetGeoInfo(ds1)
	
	ds2 	= gdal.Open(Max, GA_ReadOnly)
	
	dsOut = tifDriver.Create(OutFile, xsize, ysize, 1, Dtype)
	CopyDatasetInfo(ds1,dsOut)
	
	x_block_size = 2000#block_sizes[0]  
	y_block_size = 2000#block_sizes[1]
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
				
			A 	= ds1.GetRasterBand(Num1).ReadAsArray(j, i, cols, rows)
			B 	= ds1.GetRasterBand(Num2).ReadAsArray(j, i, cols, rows)
			C 	= ds1.GetRasterBand(Num3).ReadAsArray(j, i, cols, rows)
			D	= ds2.GetRasterBand(1).ReadAsArray(j, i, cols, rows)
			
			#Read the data into numpy arrays
			# A = BandReadAsArray(band1)
			# B = BandReadAsArray(bandMax)
			# C = BandReadAsArray(bandN1)
			# D = BandReadAsArray(bandP1)
			
			AC = numpy.subtract(A*1.0,C)
			AD = numpy.subtract(A*1.0,D)
			Bn = numpy.multiply(B,1.1)
			
			# print 'A.shape is %s' %str(A.shape)
			### Josh's suggestion
			E = numpy.zeros((rows, cols), dtype=numpy.uint8)
			E[numpy.logical_and((A >= Bn),numpy.logical_or((AD <= 20),(AC <= 20)))] = A[numpy.logical_and((A >= Bn),numpy.logical_or((AD <= 20),(AC <= 20)))]
			E[(A < B*1.1)] = A[(A < B*1.1)]

			# driver = gdal.GetDriverByName("ENVI")
			dsOut.GetRasterBand(1).WriteArray(E,j,i)
	
	# ds.GetRasterBand(1).WriteArray(NHFD1)
	#Close the datasets
	AC = None
	AD = None
	Bn = None
	A = None
	B = None
	C = None
	D = None
	ds1 = None
	ds2 = None
	dsOut = None
	
def obs_Select(bands,b,x,y,z,s):
	if b < x:
		bs 	= list(range(1,y))
		bs.remove(b)
	elif b >= bands-z:
		bs 	= list(range(bands-s,bands+1))
		bs.remove(b) 
	else:
		bs 	= list(range(b-z,b+x))
		bs.remove(b)
	return bs

def allCalc(outDir, Y_L):
	Years = Y_L.split(';')

	for Yr in Years:
		infile = glob.glob(outDir + os.sep + str(Yr)+ os.sep + "*stacked.bsq")[0]
		outputDir = outDir + os.sep + str(Yr)
		name 		= os.path.basename(infile).split('.')
		tifinfile	= gdal.Open(infile)
		cols, rows, gT, prj, dType, bands = util.GetGeoInfo(tifinfile)
		print (name[0], bands)
		fileEnvi	= outputDir + os.sep + name[0] + '_dspk.'+ name[1]
		if not os.path.exists(fileEnvi):
			## creating temprary txt file to crete vrt file to ease stacking process.
			csvFile = outputDir + os.sep + "zzzz_vrt.txt"
			# outcsv_File = open(csvFile, "wb")
			List2Stack = []
			for b in range(1,bands+1):
				print ('band %d:' %b)
				out_band = outputDir  +os.sep+ 'zzzzbnd'+str(b)+'.tif'
				if not os.path.exists(out_band):
					tempMax 	= outputDir +os.sep+ 'zzzzzMax'+str(b)+'.tif'
					# md = obs_Select(bands,b,3,6,2,4)
					me = obs_Select(int(bands),b,4,8,3,6)
					
					## computing Max value for center band with _+ 3
					tMax(infile,me,tempMax)
					
					## despiking the edge band
					if (b ==1): 
						dspike_edge(infile,tempMax,b,b+1,out_band)
					elif (b == bands):
						dspike_edge(infile,tempMax,b,b-1,out_band)
						
					## despiking all of the middle bands
					else:				
						dspike(infile,tempMax,b,b-1,b+1,out_band)
				
				List2Stack.append(out_band)
			
			util.MergeRaster(List2Stack,fileEnvi)

			for v1 in glob.glob(outputDir + os.sep+ "zzz*"):
				os.remove(v1)

def main():
	parser = OptionParser()

   # define options
	parser.add_option("-i", dest="in_Folder", help="(Required) Location of input files. e.g. ..\ddahal\eMODIS_Processing\smooths\C6")
	parser.add_option("-y", dest="Years", help="(Required) List of years")
	
	(ops, arg) = parser.parse_args()

	if len(arg) == 1:
		parser.print_help()
	elif not ops.in_Folder:
		parser.print_help()
		sys.exit(1)
	elif not ops.Years:
		parser.print_help()
		sys.exit(1)
	else:
		allCalc(ops.in_Folder,ops.Years)  

if __name__ == '__main__':

	main()
		
		
t2 = datetime.datetime.now()
print (t2.strftime("%Y-%m-%d %H:%M:%S"))
tt = t2 - t1
print ("\nProcessing time: " + str(tt) )
