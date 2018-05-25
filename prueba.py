"""import speech
import time

response = speech.input("Say something, please.")
speech.say("You said " + response)

def callback(phrase, listener):
    if phrase == "goodbye":
        listener.stoplistening()
    speech.say(phrase)

listener = speech.listenforanything(callback)
while listener.islistening():
    time.sleep(.5)"""
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave
import sys
from scipy import signal
import scipy.io.wavfile as waves
from scipy.fftpack import fft, fftfreq
def grabar(nombre):
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = nombre+".wav"

    p = pyaudio.PyAudio()

    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = chunk)

    print "* recording"
    all = []
    for i in range(0, RATE / chunk * RECORD_SECONDS):
        data = stream.read(chunk)
        all.append(data)
    print "* done recording"

    stream.close()
    p.terminate()

    # write data to WAVE file
    data = ''.join(all)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

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

grafica("uno_ar.wav",'t')
grafica("uno_ax.wav",'t')

grafica("hola.wav",'f')

"""
BASE DE CONOCIMIENTO:
    paso 2
    segmentar en 4
    ventanear cada segmento
    hacemos vector caracteristico de cada segmento (amp, frec)
RED:
definir tam de la RED
entrada : vec_carac_de_la_PRUEBA
pesos:aleatorio o en ceros



"""