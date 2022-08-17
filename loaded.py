#Always make all imports in the first cell of the notebook, run them all once.
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from skimage import feature
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
#from __future__ import division
from scipy.signal import convolve2d
from sklearn.neighbors import KNeighborsClassifier
import time as time
import sys
import pickle

test_dir = sys.argv[1]
out_dir = sys.argv[2] 

def preprocess(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(3,3))    # blur the image to remove the noise
    thresh, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    number_of_white_pix = np.sum(bin_img == 255)
    number_of_black_pix = np.sum(bin_img == 0)


    if number_of_white_pix > number_of_black_pix:
        img = 255 - img
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray,(3,3)) 
        thresh, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #resized_img = cv2.resize(bin_img, (330, 120))
    return bin_img


def lpq(img,winSize=3,freqestim=1,mode='nh'):
    rho=0.90

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA=8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
        w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode=='im':
        LPQdesc=np.uint8(LPQdesc)

    ## Histogram if needed
    if mode=='nh' or mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode=='nh':
        LPQdesc=LPQdesc/LPQdesc.sum()

    #print(LPQdesc.shape)
    return LPQdesc

test_sub = []
for filename in sorted(glob.glob( test_dir + '*.png')):
    img = cv2.imread(filename) ## cv2.imread reads images in JPG format
    test_sub.append(img)

test_sub_preprocessed = []

for i in range(len(test_sub)):
    test_sub_preprocessed.append(preprocess(test_sub[i]))

lpq_sub = []

for i in range(len(test_sub_preprocessed)):
    lpq_sub.append(lpq(test_sub_preprocessed[i], mode='nh'))

lpq_classifier = pickle.load(open("lpq_classifier.txt", 'rb'))

y_sub = lpq_classifier.predict(lpq_sub)

file1 = open(out_dir + 'results.txt',"w+")
file2 = open(out_dir + 'times.txt',"w+")
for i in range(len(test_sub)):
    start_time = time.time()
    im_sub = preprocess(test_sub[i])
    lpq_sub_img = lpq(im_sub)
    lpq_sub_img = [lpq_sub_img]
    sub_pred = lpq_classifier.predict(lpq_sub_img)
    file1.write(str(sub_pred[0]) + "\n")
    end_time =time.time() - start_time 
    end_time = round(end_time , 3)
    if end_time == 0 :
        end_time = 0.001 
    file2.write(str(end_time) + "\n")
file1.close()
file2.close()