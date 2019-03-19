'''
Date: 3/8/2019
Author: Devendra Dahal
Update: 
Description: This script is developed to validate predicted NDVI by computing stats and plotting 
			boxplots by landcover classes for each week. This is part of quality controling.
USAGE: 
'''

import ProjRaster as prs
import utility_27 as util
import os, sys , time, datetime
from glob import glob
import numpy as np
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
from optparse import OptionParser 
import seaborn as sns
import matplotlib.pyplot as plt
try:
	from osgeo import gdal
	from osgeo.gdalconst import *
	from osgeo.gdalnumeric import *
	from osgeo import ogr
	from osgeo import osr
	from osgeo import gdal_array
except ImportError:
	import gdal

t1 = datetime.datetime.now()
print (t1.strftime("%Y-%m-%d %H:%M:%S"))

# Number of Workers.  This can be larger depending CPU of the system
pool = Pool(5) 

# define worker function before a Pool is initiated
def worker(NewProcess):
	try:
		api.my_operation(NewProcess)
	except:
		print('error with item')

tifDriver = gdal.GetDriverByName('GTiff')
tifDriver.Register()

def RasterSave(CArray,OutFile,cMap,co, ro,bnd):
	dsOut = tifDriver.Create(OutFile, co, ro, bnd, gdal.GDT_Byte)
	CopyDatasetInfo(cMap,dsOut)
	bandOut=dsOut.GetRasterBand(bnd)
	BandWriteArray(bandOut, CArray)
	bandOut=None
	
def ReClassNLCD(inFile,outFile):
	'''This function reclassify NLCD forest classes and all rangeland classes to just two and 
		all other classes will be nodata'''

	nRas= gdal.Open(inFile)
	xsize, ysize, gT, prj, dType, bnds = util.GetGeoInfo(nRas)

	dsOut = tifDriver.Create(outFile, xsize, ysize, 1, dType)
	gdal_array.CopyDatasetInfo(nRas,dsOut)
	# bandOut=dsOut.GetRasterBand(1)
	#preRas = None
	#nxtRas = None

	x_block_size = 3660#block_sizes[0]  
	y_block_size = 3660#block_sizes[1]
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
			
			inRas =nRas.GetRasterBand(1).ReadAsArray(j, i, cols, rows)
			
			inRas = inRas.astype(float)
			
			newRast = np.where((inRas > 40) &(inRas < 45),10,np.where((inRas == 52) | (inRas == 71),20, np.nan))
			newRast[(newRast == 0)] = np.nan
			
			dsOut.GetRasterBand(1).WriteArray(newRast,j,i)
			inRas= None
	nRas = None
	newRast = None
	dsOut = None
	
def CalcAll(inFolder, NLCDDir, OutputFolder):
	
	if not os.path.exists(OutputFolder):
		os.makedirs(OutputFolder)
		
	Yrs = glob(inFolder +os.sep+'201*')
	
	# di = {20: 'Rangeland',10:'Forest'}
	for yr in Yrs:
		Year = os.path.basename(yr)
		print (' ')
		print (Year)
		
		for tile in glob(yr +os.sep+'1*'):
			# tile = yr +os.sep+tileName
			tileName = os.path.basename(tile)
			print (tileName)
			
			
			FigDir = OutputFolder.replace('Datas','Figures')
			# outFile = FigDir + os.sep + tileName+'_'+str(Year) +'_InterpolatedNDVI_boxplot.png'
			outFile = FigDir + os.sep + tileName+'_'+str(Year) +'_InterpolatedNDVI_boxplot.png'
			if not os.path.exists(outFile):
				
				ListRasAll_Fill = sorted(glob(tile +os.sep+ "*new*.tif"))
				"""
				## Creating list of predicted NDVI maps
				ListRasAll_Fill = []
				for wk in sorted(glob(tile +os.sep+'wk*')):
					WEEK = os.path.basename(wk)
					wkRas = wk+os.sep+tileName+'_'+str(Year)+'_'+WEEK+'_Cubist.tif'
					if os.path.exists(wkRas):
						ListRasAll_Fill.append(wkRas)
					else:
						print('Raster for %s is not predicted.' %wkRas)
						sys.exit(1)
				"""
				RefFile = NLCDDir + os.sep + 'HLS_T'+tileName + '_NLCD2011_2Cls.tif'
				if not os.path.exists(RefFile):
					NLCD = NLCDDir + os.sep + 'HLS_T'+tileName + '_NLCD2011_UTM.tif'
					
					ReClassNLCD(NLCD,RefFile)
				
				SampleCSV = OutputFolder + os.sep + tileName+'_'+str(Year) +'_samplezzz.csv'
				if not os.path.exists(SampleCSV):
					util.stratifed_random(RefFile,SampleCSV,5000)
				
				util.CleanCSV(SampleCSV, 'cell', 0)
				
				HeaderList = ['Class','x','y']
				for i in ListRasAll_Fill[2:-2]:
					outName = os.path.basename(i).split('.')[0]
					Nbreak = outName.split("_")
					# if not Nbreak[1] == 'T'+BName:
					# date = 'd'+Nbreak[3][-3:]
					Wk = Nbreak[2]
					HeaderList.append(Wk)
				outputfile = OutputFolder + os.sep + tileName +'_'+ str(Year) +'_allNDVImerged.tif'
				if not os.path.exists(outputfile):
					util.MergeRaster(ListRasAll_Fill[2:-2],outputfile)
				'''------Image value extraction----- start here'''

				# RanPntCsv = OutputFolder + os.sep + outName+'.data'
				OutFinal = OutputFolder + os.sep + tileName+'_'+str(Year) +'_QualityC.csv'
				if not os.path.exists(OutFinal):
					print ('\nextracting values .....')
					util.extValue(outputfile,SampleCSV,OutFinal, HeaderList,'trend')
				
				### Plot graphics (boxplot and swarmplots)
				
				
				inCsv = pd.read_csv(OutFinal, low_memory=False)
				
				# inCsv = inCsv.iloc[:,'wk05':'wk44'] > 100
				# inCsv = inCsv.iloc[:,'wk05':'wk44'] <= 200
				for col in range(5,41):
					wk = 'wk'+str(col+2).zfill(2)
					inCsv = inCsv.loc[(inCsv[wk] > 100) & (inCsv[wk] <= 200)] 
				
				newTable =  pd.DataFrame(columns = ['LClass','week','NDVI'])
				for j in (10,20):
					for i in range(5,inCsv.shape[1]):
						wk = 'wk'+str(i+2).zfill(2)
						lc = inCsv.loc[inCsv['Class'] == j, 'Class']
						week = pd.DataFrame(index=np.arange(lc.shape[0]), columns=np.arange(1))
						week[0] = i+2
						NDVI = pd.Series.to_frame(inCsv.loc[inCsv['Class'] == j, wk])
						result =  pd.concat([lc.reset_index(drop=True), week.reset_index(drop=True),NDVI.reset_index(drop=True)], axis=1)
						result.columns = ['LClass', 'week','NDVI']
						newTable = pd.concat([newTable.reset_index(drop=True),result], axis=0)

				newTable.LClass.replace([10, 20], ['Forest', 'Rangeland'], inplace=True)
				# newTable = newTable[newTable['LClass'] == 20.0] = di[20]
				plt.figure(figsize = (18,7))

				sns.set(font_scale = 1.5)
				sns.set_style('ticks')

				ax = sns.boxplot(x = 'week', y= 'NDVI',
							data=newTable, 
							hue = 'LClass',
							fliersize = 0.2, 
							palette="Set3")
				ax.tick_params(
						axis='both',          # changes apply to the both axis
						which='both',      # both major and minor ticks are affected
						top=False,         # ticks along the top edge are off
						right=False,         # ticks along the right edge are off
						labelbottom=True # labels along the bottom edge
						)
				ax.legend(loc='upper right')
				ax.set_xticklabels(range(7,41), rotation =45)
				ax.set(xlabel='Weeks',ylabel='NDVI')
				ax.set_title('Interpolated (gap-filled) NDVI for '+tileName+' '+ Year)
				# ax.set_title('Predicted NDVI for '+tileName+' '+ Year)
				ax.figure.savefig(outFile)
				plt.close()
				# ax.close()
			for z in glob(OutputFolder + os.sep + '*zz*'):
				os.remove(z)
			# sys.exit()

def main():
	parser = OptionParser()

   # define options
	parser.add_option("-i", dest="inFolder", help="(Required) Location to input data")
	parser.add_option("-n", dest="inNLCD", help="(Required) Location of NLCD data")
	parser.add_option("-o", dest ="outFolder", help="(Required) Location to save output data")
	
	(ops, arg) = parser.parse_args()

	if len(arg) == 1:
		parser.print_help()
	elif not ops.inFolder:
		parser.print_help()
		sys.exit(1)
	elif not ops.inNLCD:
		parser.print_help()
		sys.exit(1)
	elif not ops.outFolder:
		parser.print_help()
		sys.exit(1)
	else:
		pool.apply_async(worker,CalcAll(ops.inFolder,ops.inNLCD, ops.outFolder))

if __name__ == '__main__':

	main()

pool.close()
pool.join()

t2 = datetime.datetime.now()
print (t2.strftime("%Y-%m-%d %H:%M:%S"))
tt = t2 - t1
print ("\nProcessing time: " + str(tt)) 
print ("done!! I am pretty sure\n\n")