# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:06:46 2023

@author: 97345
"""

import prosail
import numpy as np
#   Input the parameter file and spectral response
inFile = r'F:\01shiyan\ANN\input_prosail.txt'

result = np.zeros(shape=(2101,10001))
inputData=open(inFile,'r')

row = 0
#   Greater than or equal to 400, less than 2500nm, to achieve the first column of storage wavelength information
for lam in range(400,2501):
    result[row,0] = lam
    row = row + 1

#   Input the parameters
c = 1
j = 0
for lines in inputData:
    s=[]
    ls = lines.split('\t')
    for i in ls:
        s.append(i)
    j = j + 1
    if j>1:
        cab = float(s[4])
#        Cxc(Car)：
        car = float(s[6])
        cbrown = 0.0
        cw = float(s[2])
        cm = float(s[5])
        ant = float(s[7])
        n = float(s[3])
        
        lai = float(s[0])
        lidfa = float(s[1])
        hspot = 0.5/lai
        psoil = float(s[8])
        
        tts = 30.0
        tto = 0.0   
        psi = 90.0
        # Run prosail (Prosail-D and sail) models to generate spectral reflectance simulation curves
        # ant: anthocyanin; typelidf: Type of lobe Angle (2: average leaf inclination); factor: directional reflectance; psoil: soil reflectance coefficient
        rho_canopy=prosail.run_prosail(n, cab, car, cbrown, cw, cm, lai, lidfa, hspot, tts, tto, \
                                       psi, ant, alpha=40.0, prospect_version='D', typelidf=2, lidfb=0.0, \
                                       factor='SDR', rsoil0=None, rsoil=1, psoil=psoil, \
                                       soil_spectrum1=None, soil_spectrum2=None)
        l = 0
        # The results of different bands
        for i in range(len(rho_canopy)):
            #Starting from row 1, column 2, all the way to 2500nm
            result[l,c] = rho_canopy[i]
            l = l + 1
        c= c + 1

inputData.close()

#Write a file to store the results of the prosail simulation
simulationFile= r'F:\01shiyan\ANN\prosail_multiSimulation.txt'

#Save the result to a simulationFile
np.savetxt(simulationFile,result,fmt="%.6f",delimiter=" ")

#   SIMLAB format
formatProsail = r'E:\02shiyan\PROSAIL_Test\simlab_sentinel\Prosail_format.txt'

with open(formatProsail,mode='a') as format:
    format.write('1')
    format.write('\n')
    format.write('SIMREF')
    format.write('\n')
    format.write('time = yes')
    format.write('\n')  
    format.write('4991')
    format.write('\n')
    for c in range(1,4992):
        simRef = result[:,c]
        format.write('RUN    '+str(c-1))
        format.write('\n')
        format.write('2101')
        format.write('\n')
        for v in range(0,2101):
            format.write(str(400+v)+' '+str("%6f"%simRef[v]))
            format.write('\n')



