import math
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave
import sys
from scipy import signal
import scipy.io.wavfile as waves
import time
from scipy.fftpack import fft, fftfreq


#####       MFCC    ###########################

import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, IPython.display


#########################################################################
###############  BASE DE CONOCIMIENTO VOCALES   #########################


x, fs = librosa.load('a_axel1.wav')
x2, fs2 = librosa.load('a_axel2.wav')
x3, fs3 = librosa.load('a_axel3.wav')
x4, fs4 = librosa.load('a_axel4.wav')
x5, fs5 = librosa.load('a_axel5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)

promedio_a_axel=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5
x, fs = librosa.load('a.wav')
x2, fs2 = librosa.load('a2.wav')
x3, fs3 = librosa.load('a3.wav')
x4, fs4 = librosa.load('a4.wav')
x5, fs5 = librosa.load('a5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)

promedio_a_arturo=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5
#print promedio_a_arturo

x, fs = librosa.load('e_axel1.wav')
x2, fs2 = librosa.load('e_axel2.wav')
x3, fs3 = librosa.load('e_axel3.wav')
x4, fs4 = librosa.load('e_axel4.wav')
x5, fs5 = librosa.load('e_axel5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)


promedio_e_axel=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5
#print promedio_e_arturo
x, fs = librosa.load('e_ar.wav')
x2, fs2 = librosa.load('e_ar2.wav')
x3, fs3 = librosa.load('e_ar3.wav')
x4, fs4 = librosa.load('e_ar4.wav')
x5, fs5 = librosa.load('e_ar5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)

promedio_e_arturo=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5


x, fs = librosa.load('i_axel1.wav')
x2, fs2 = librosa.load('i_axel2.wav')
x3, fs3 = librosa.load('i_axel3.wav')
x4, fs4 = librosa.load('i_axel4.wav')
x5, fs5 = librosa.load('i_axel5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)


promedio_i_axel=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5
#print promedio_i_arturo
x, fs = librosa.load('i_ar.wav')
x2, fs2 = librosa.load('i_ar2.wav')
x3, fs3 = librosa.load('i_ar3.wav')
x4, fs4 = librosa.load('i_ar4.wav')
x5, fs5 = librosa.load('i_ar5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)

promedio_i_arturo=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5


x, fs = librosa.load('o_axel1.wav')
x2, fs2 = librosa.load('o_axel2.wav')
x3, fs3 = librosa.load('o_axel3.wav')
x4, fs4 = librosa.load('o_axel4.wav')
x5, fs5 = librosa.load('o_axel5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)


promedio_o_axel=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5
#print promedio_o_arturo
x, fs = librosa.load('o_ar.wav')
x2, fs2 = librosa.load('o_ar2.wav')
x3, fs3 = librosa.load('o_ar3.wav')
x4, fs4 = librosa.load('o_ar4.wav')
x5, fs5 = librosa.load('o_ar5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)

promedio_o_arturo=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5


x, fs = librosa.load('u_axel1.wav')
x2, fs2 = librosa.load('u_axel2.wav')
x3, fs3 = librosa.load('u_axel3.wav')
x4, fs4 = librosa.load('u_axel4.wav')
x5, fs5 = librosa.load('u_axel5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)


promedio_u_axel=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5
#print promedio_u_arturo
x, fs = librosa.load('u_ar.wav')
x2, fs2 = librosa.load('u_ar2.wav')
x3, fs3 = librosa.load('u_ar3.wav')
x4, fs4 = librosa.load('u_ar4.wav')
x5, fs5 = librosa.load('u_ar5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)

promedio_u_arturo=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5




promedio_a= (promedio_a_axel + promedio_a_arturo) / 2
promedio_e= (promedio_e_axel + promedio_e_arturo) / 2
promedio_i= (promedio_i_axel + promedio_i_arturo) / 2
promedio_o= (promedio_o_axel + promedio_o_arturo) / 2
promedio_u= (promedio_u_axel + promedio_u_arturo) / 2



## ciclo de comparacion del error minimo

lista_promedios=[]
lista_vocales=['a','e','i','o','u']
lista_error=[]

lista_promedios.append(promedio_a)
lista_promedios.append(promedio_e)
lista_promedios.append(promedio_i)
lista_promedios.append(promedio_o)
lista_promedios.append(promedio_u)





































"""

#######################   VOCAL DE PRUEBA 'O' DE AXEL
x, fs = librosa.load('o1.wav')
mfccs1 = librosa.feature.mfcc(x, sr=fs)
promedio_o_axel=(abs(sum(sum(mfccs1))))

for i in lista_promedios:
    error=promedio_o_axel-i
    lista_error.append(abs(error))

posicion = lista_error.index(min(lista_error))

print "la vocal que se dijo fue: ", lista_vocales[posicion]






x, fs = librosa.load('u_prueba.wav')
mfccs = librosa.feature.mfcc(x, sr=fs)

vocal_a=abs(sum(sum(mfccs)))

lista_a=[]
for i in range(5):
    error= vocal_a-lista_promedios[i]
    lista_a.append(abs(error))
posicion = lista_a.index(min(lista_a))

print "la vocal que se dijo fue: ", lista_vocales[posicion]

"""


