# I used code from HW8 as well as some from HW3 
# I also borrowed Audio Utilities to read wav files in
# To use the program just make a call to the function main with the 
# filename which will read the file then perform onset detection  
# and then auto correlate those onsets and then find a peak based on the 
# threshold used. It will then print out the array of tuples which contain the beat 
# detected for that window as well as the # of seconds at which it occurs.
# Prints graphs of each song's tempo(BPM / Time in seconds)
# You can change the window size of the sample and the threshold 

import numpy as np
import matplotlib.pyplot as plt
import math
import audioUtilities as au

def maxabs(Y):
   max = Y[0]
   for i in range(len(Y)):
       if(abs(Y[i]) > max):
          max = abs(Y[i])
   return max
   
   
def getEnvelope(X,W,S):
   N = math.ceil(len(X)/S)
   E = [0]*N
   for i in range(N):
       w = i*S 
       E[i] = maxabs(X[w:(w+W)])
   return E

def realFFT(X):
   return [2.0 * np.absolute(x)/len(X) for x in np.fft.rfft(X)]


def findPeaks(X, tempo, start, threshold):
  for i in range(1, len(X)):
    if i == len(X) - 1:
        break
    if X[i] < 0:
        continue
    if X[i] > threshold and X[i] > X[i-1] and X[i] > X[i+1]:
      #Append a tuple so we know where in the window it occurs and break out
      tempo.append((i+start, i))
      break


def autoCorrelation(X):
  auto_correlation = []
  bound = int(math.floor(len(X)/2))
  for lag in range(int(bound)):
    total = 0
    if lag == 0:
      continue
    for i in range(len(X)-lag):
      total += X[i] * X[i+lag]
    auto_correlation.append((total / (len(X) - lag)))

  return auto_correlation

def findTempo(fileName):

  sample = au.readWaveFile(fileName)

  #print(len(sample))

  tempo = []

  # Loop through desired window size; Used 441000(10secs) 
  # Sliding (non overlapping)

  for i in range(0, len(sample), 441000):
    X = sample[i:i+441000]

    #print(len(X))

    B = [i**2 for i in X]

    A = getEnvelope(B,1024,1000)

    D = np.diff(A)

    D = [i if i > 0 else 0 for i in D ]

    S = realFFT(D)

    #using same threshold as in hw8 can change from 30% to anything higher or lower

    threshold = max(S)*.3

    auto = autoCorrelation(S)

    #print(auto)
    #print(len(auto))

    #beat = 60*i*44100/len(sample)
    
    findPeaks(auto, tempo, 0+i, threshold)

    #divide x by 44100 to figure out how many seconds into the song we are
  return [(60*z*44100/441000, x/44100) for (x, z) in tempo]

def main(fileName):
  X = findTempo(fileName)
  py = [y for (y, x) in X]
  px = [x for (y, x) in X]
  plt.plot(px, py)
  plt.title(fileName)
  plt.ylabel("BPM")
  plt.xlabel("Seconds")
  plt.show()
  print(X)
  


#main("Demi Lovato Catch Me Piano.wav")
#main("Taylor Swift You Belong With Me.wav")
#main("Imagine Dragons Demons.wav")
#main("J.J Cale Call Me The Breeze.wav")