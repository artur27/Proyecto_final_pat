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


def grafica(segnal,tof):
    #grabar("hola")
    muestreo, sonido = waves.read(segnal)
    L = len(sonido)
    freq = np.fft.fftfreq(L)
    freq_in_hertz = abs(freq * muestreo)
    #signal.spectrogram(muestreo,sonido)
    print muestreo
    #plt.plot(sonido)
    #plt.show()

    Yc = np.fft.fft(sonido)
    #plt.plot(freq_in_hertz,abs(Yc))
    if tof=='t':
        plt.plot(sonido)
    else:
        plt.plot(freq_in_hertz,abs(Yc))
    plt.show()

#grafica("uno_ar.wav",'t')
#grafica("uno_ax.wav",'t')
#grafica("hola.wav",'f')
frec1,data1=waves.read("o1.wav")
frec2,data2=waves.read("o2.wav")
frec3,data3=waves.read("o3.wav")
frec4,data4=waves.read("o4.wav")
frec5,data5=waves.read("o5.wav")








### EXTRACCION DE CARACTERISTICAS
import numpy
from python_speech_features import sigproc
from scipy.fftpack import dct
def hz2mel(hz):

    return 2595 * numpy.log10(1+hz/700.)
def mel2hz(mel):

    return 700*(10**(mel/2595.0)-1)
def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:numpy.ones((x,))):

    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat
def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          winfunc=lambda x:numpy.ones((x,))):

    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    energy = numpy.sum(pspec,1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy
def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):

    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank





#normalizacion
points1 = len(data1)
maximo1 = max(data1[0:points1])

data1=data1/float(maximo1)
timeArray1 = np.arange(0,points1,1)
timeArray1 = timeArray1/float(frec1)

points2 = len(data2)
maximo2 = max(data2[0:points2])

data2=data2/float(maximo2)
timeArray2 = np.arange(0,points2,1)
timeArray2 = timeArray2/float(frec2)

#transformada
F1= fft(data1[0:points1])
N=len(timeArray1)
dt = 1/float(frec1)
w=fftfreq(N,dt)
#plt.plot(w,abs(F1),'b-')
#plt.show()
 

F2= fft(data2[0:points2])
N2=len(timeArray2)
dt2 = 1/float(frec2)
w2=fftfreq(N2,dt2)
#plt.plot(w2,abs(F2),'b-')
#plt.show()

#Frecuencias que no necesitamos
lim1=-1000
lim2=-80
lim3= lim1*-1
lim4= lim2*-1
freqs=w

for i in range(len(F1)):
    if w[i]<lim1:
        F1[i]=0
    elif w[i]>lim2 and w[i]<lim4:
        F1[i]=0
    elif w[i]>lim3:
        F1[i]=0
#filtro
b,a= signal.butter(10,0.125,btype='highpass')
fil1=signal.filtfilt(b,a,abs(F1),0)
#plt.plot(w,abs(F1))
#plt.show()
# VETANEO

caracteristicas = []
ventana= np.hamming(320)
for i in range(frec1/320):
    caracteristicas.append(ventana*w[(i)*320:(i+1)*320])

valores1,a1= fbank(caracteristicas[0],16000,0.025,0.01,26,512,0,None,0.97)
print
print
valores2,a2= fbank(caracteristicas[1],16000,0.025,0.01,26,512,0,None,0.97)
print 




#####       MFCC    ###########################

import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, IPython.display


#########################################################################
###############  BASE DE CONOCIMIENTO VOCALES   #########################



#print promedio_o_axel

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
#print promedio_e_arturo


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
#print promedio_i_arturo



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
#print promedio_o_arturo



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
#print promedio_u_arturo


#######################   VOCAL DE PRUEBA 'O' DE AXEL
x, fs = librosa.load('o1.wav')
x2, fs2 = librosa.load('o2.wav')
x3, fs3 = librosa.load('o3.wav')
x4, fs4 = librosa.load('o4.wav')
x5, fs5 = librosa.load('o5.wav')

mfccs1 = librosa.feature.mfcc(x, sr=fs)
mfccs2 = librosa.feature.mfcc(x2, sr=fs2)
mfccs3 = librosa.feature.mfcc(x3, sr=fs3)
mfccs4 = librosa.feature.mfcc(x4, sr=fs4)
mfccs5 = librosa.feature.mfcc(x5, sr=fs5)

promedio_o_axel=(abs(sum(sum(mfccs1)))+ abs(sum(sum(mfccs2))) + abs(sum(sum(mfccs3)))+ abs(sum(sum(mfccs4))) +abs(sum(sum(mfccs5))))/5




x, fs = librosa.load('i_prueba.wav')
mfccs = librosa.feature.mfcc(x, sr=fs)

vocal_a=abs(sum(sum(mfccs)))
print vocal_a

## ciclo de comparacion del error minimo

lista_promedios=[]
lista_vocales=['a','e','i','o','u']
lista_error=[]
lista_promedios.append(promedio_a_arturo)
lista_promedios.append(promedio_e_arturo)
lista_promedios.append(promedio_i_arturo)
lista_promedios.append(promedio_o_arturo)
lista_promedios.append(promedio_u_arturo)
for i in lista_promedios:
    error=promedio_o_axel-i
    lista_error.append(error)

posicion = lista_error.index(max(lista_error))
print posicion
print "la vocal que se dijo fue: ", lista_vocales[posicion]
print lista_promedios
lista_a=[]
for i in range(5):
    error= vocal_a-lista_promedios[i]
    lista_a.append(abs(error))
posicion = lista_a.index(min(lista_a))
print posicion
print "la vocal que se dijo fue: ", lista_vocales[posicion]

# otra manera de obtener los coeficientes pero que solo nos devuelva 40 valores
#y, sr = librosa.load('o1.wav')
#mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)








"""
print maximo
data = data/float(maximo)
BASE DE CONOCIMIENTO:
    paso 2
    segmentar en 4
    ventanear cada segmento
    hacemos vector caracteristico de cada segmento (amp, frec)
RED:
definir tam de la RED
entrada : vec_carac_de_la_PRUEBA
pesos:aleatorio o en ceros


#w = abs(w)
#w=math.sqrt(w)

#Modulo de W

modu=w.real**2+w.imag**2
#print modu

for i in range(len(w)-1):
    if modu[i] == 0:
        modu[i] = 0.001
    else:
        w[i]=math.log(math.sqrt(modu[i]))

"""
