# _*_ coding: utf-8 _*_
import numpy as np
import gdal
from osgeo import gdal
from osgeo.gdalconst import *
import glob
from sklearn.externals import joblib
from sklearn import preprocessing

# image processing
class GRID:
    # load image
    def read_img(self,filename):
        dataset=gdal.Open(filename)
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_geotrans = dataset.GetGeoTransform()
        im_proj = dataset.GetProjection()
        im_data = dataset.ReadAsArray(0,0,im_width,im_height)
        del dataset
        return im_proj,im_geotrans,im_data
    
    # write image
    def write_img(self,filename,im_proj,im_geotrans,im_data):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

        del dataset

if __name__ == "__main__":
    run = GRID()
    proj, geotrans, data = run.read_img(r'F:\01shiyan\reshamp\bound\B2.tif')
    in_ds = gdal.Open(r'F:\01shiyan\reshamp\bound\B2.tif')
    in_band = in_ds.GetRasterBand(1)
    xsize = in_band.XSize
    ysize = in_band.YSize
    
    # output
    out_ds = in_ds.GetDriver().Create('LAI.tif', xsize, ysize, 1, gdal.GDT_Float32) 
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    
####################
    
    fileList = glob.glob(r'F:\01shiyan\reshamp\bound\B*.tif')
    num = len(fileList)
    i = 0
    
    datas = np.zeros([ysize,xsize,9])
    for file in fileList:  
        print(file)
        dataset = gdal.Open(file,GA_ReadOnly)
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray(0,0,xsize,ysize)
        datas[:,:,i] = data  
        i = i + 1
    datas = datas * 0.0001   
    datas[datas<=0.0]=0.0001
    datas[datas>=1.0]=1.0
    
    #############
    inputData = np.reshape(datas,(ysize*xsize,9)) 
    mms = preprocessing.MinMaxScaler()
    inputData=mms.fit_transform(inputData)
    mlp_model = joblib.load(filename='ANN.model')  
    outputData = mlp_model.predict(inputData)  
    #############
    

####################
    
    minLAI = min(outputData)
    maxLAI = max(outputData)
    k = 8 / (maxLAI - minLAI)
    for i in range(0, len(outputData)):
        if outputData[i] == 0:
            outputData[i] = outputData[i]
        else:
            outputData[i] = k * (outputData[i] - minLAI)

    outMatrix = np.reshape(outputData, (ysize, xsize))  
    out_band.WriteArray(outMatrix, 0, 0)

    out_band.FlushCache()  
    out_ds = None