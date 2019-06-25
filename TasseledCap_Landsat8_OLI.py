def tasseled_cap(in_ds,  rows, cols, j, i):

		"""
    		Function to calculate Tasseled cap for Landsat 8 (OLI) using numpy array blocks.
    
		Ali Baig, Muhammad Asad et al. (2014). Derivation of a tasselled cap transformation based on Landsat 8
			at-satellite reflectance. Remote Sensing Letters 5(5):423-431.

			Bands 2, 3, 4, 5, 6, 7
		"""
		
		tcap_coefficients = np.array([(0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872), 
									(-0.2941, -0.243, -0.5424, 0.7276, 0.0713,-0.1608), 
									(0.1511, 0.1973, 0.3283,0.3407, -0.7117, -0.4559), 
									(-0.8239,0.0849, 0.4396, -0.058, 0.2013, -0.2773), 
									(-0.3294, 0.0557, 0.1056, 0.1855, -0.4349, 0.8085), 
									(0.1079, -0.9023, 0.4119, 0.0575,  -0.0259, 0.0252)], dtype='float32')
		
		newArray = in_ds.ReadAsArray(j, i, cols, rows)
		shp = newArray.shape

		newArray = newArray.reshape(shp[0], shp[1]*shp[2])

		tcap_array = np.dot(tcap_coefficients, newArray).reshape(shp)
		
		### For all 6 TC bands, comment out line below and change band number to 6 when saving as out raster
		tcap_array = tcap_array[:3]

		return tcap_array
