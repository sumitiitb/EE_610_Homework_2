'''
Author          : Sumit Kumar
Functionality   : Image De-blur
Repository      : Github (https://github.com/sumitiitb/EE_610_Homework_2)

---------------------------------------------------------------------------
                  Disclaimer !! 
---------------------------------------------------------------------------
This code has been written towards partial fulfillment of Assignment 2 of EE 610.
All the lines of this code is self written.
However, some of the technical concepts have been learnt from various sources (Internet). 
These sources (along with the concept that was learnt) have been mentioned as and when relevant

A part of the code and the GUI has been taken from the Assignment 1 (of EE 610)


'''
#Import Necessary Libraries

import sys   
import cv2
from PyQt4 import QtGui,QtCore,Qt
from matplotlib import pyplot as plt	
from PyQt4.Qt import *	
import numpy as np	
import math as mt
import skimage as skm
import cmath

app = QtGui.QApplication(sys.argv)          #Instantiation of QT Applciation
wind = QtGui.QWidget()                      #Instantiation of QT widget. Widget holds all the buttons, levels etc.
wind.setGeometry(10,50,1250,640)            #setting the dimensions of the User Interface


inlabel = QtGui.QLabel(wind)                #Instantiating label to display input image
inlabel.setText("Blur Input Image")         #Text to be displayed in input label
inlabel.setGeometry(40,30,350,350)          #setting the dimensions of input label. The image will be displayed according to this size


k_label = QtGui.QLabel(wind)                #Instantiating label to display output image
k_label.setText("Kernel")                   #Text to be displayed in output label
k_label.setGeometry(410,30,41,41)           #setting the dimensions of input label. The image will be displayed according to this size

outlabel = QtGui.QLabel(wind)               #Instantiating label to display output image
outlabel.setText("Reference Image")         #Text to be displayed in output label
outlabel.setGeometry(740,30,350,350)        #setting the dimensions of input label. The image will be displayed according to this size

img = np.array(object)                      #class varidisplay_ref_imageable for input image (for reading input image from file)
outimg = np.array(object)                   #class variable for output image (for passing output image between various functions)
in_img = np.array(object)                   #class variable for input image (for passing input image between various functions)
recent_outimg = np.array(object)            #class variable for output image (for Undo Last)
mask = np.array(object)


#This function loads input image. In this case input image is a blurry image.
# slot for load image button
def on_load_button(self):
    global img,outimg,in_img,recent_outimg,mask,input_image_copy,global_input_img_hsv   #global variables
    filename = QtGui.QFileDialog.getOpenFileName()      #open the file browser to select an image
    if not filename:
        print("unable to load image")                   #check if the filename is valid, else return
        return
    img = cv2.imread(filename,0)                        #now read the image from harddisk, the second argument '0' means load a grayscale image
    #img  = np.float32(img)/255
    input_image_copy = cv2.imread(filename)             #read the input image from harddisk
    print("Original Image Shape : ")    
    print(input_image_copy.shape)
    '''
    m,n = img.shape
    kernel = np.tile(img,(19,19))
    kernel = np.pad(kernel, [(0, 800 - kernel.shape[0]), (0, 800 - kernel.shape[1])], 'constant')  
    temp_img = kernel
     #fh_ishift = fb_shift    
    for i in range(0,800):                              
        for j in range(0,800):
            if((i< 370) or (i > 410)or (j< 370 )or(j>410)):
                temp_img[i,j] = 0   
    print("Temp kernel created")
    cv2.imwrite('temp_center_kernel.jpg',temp_img)  
    '''
    input_img_hsv = cv2.cvtColor(input_image_copy, cv2.COLOR_BGR2HSV) #conver the input image into hsv format
    channel_h, channel_s, channel_v = cv2.split(input_img_hsv)        #split the input image to get different parts of the image
    in_img = channel_v                                                #assign channel_v to variable in_img      
    cv2.imwrite('temp_input.jpg',input_image_copy)                    #creating a temp file so as to keep a backup of input image  
    global_input_img_hsv = cv2.cvtColor(input_image_copy, cv2.COLOR_BGR2HSV) #creating a global hsv image
    if img is None:
        print("unable to load image %s"%(filename))     #checking if the input image is correctly loaded or not
        return
    print("image file %s is opened"%(filename))         #print the name and flush output
    display_input_image()                               #function to actually update the input image in GUI
    outimg = in_img                                     #passing the input image into variable outimg. outimg variable is a global variable
    recent_outimg = outimg                              #passing the outimg value to another variable to keep track of the last outimg (useful for undo last)



#This function loads input image. In this case input image is a blur kernel image.
# slot for load image button
def load_blur_kernel(self):
    global blur_img   #global variables
    filename = QtGui.QFileDialog.getOpenFileName()      #open the file browser to select an image
    if not filename:
        print("unable to load image")                   #check if the filename is valid, else return
        return
    blur_img = cv2.imread(filename,0)                        #now read the image from harddisk, the second argument '0' means load a grayscale image
    blur_img = np.float32(blur_img)/255
    if blur_img is None:
        print("unable to load image %s"%(filename))     #checking if the input image is correctly loaded or not
        return
    print("image file %s is opened"%(filename))         #print the name and flush output
    display_blur_image()                               #function to actually update the input image in GUI

#This function loads input image. In this case input image is a reference image (or Groundtruth)
# slot for load reference image button
def load_ref_image(self):
    global ref_img   #global variables
    filename = QtGui.QFileDialog.getOpenFileName()      #open the file browser to select an image
    if not filename:
        print("unable to load image")                   #check if the filename is valid, else return
        return
    ref_img = cv2.imread(filename)                        #now read the image from harddisk, the second argument '0' means load a grayscale image
    
    if ref_img is None:
        print("unable to load image %s"%(filename))     #checking if the input image is correctly loaded or not
        return
    print("image file %s is opened"%(filename))         #print the name and flush output
    cv2.imwrite('temp_ref_input.jpg',ref_img)  
    display_ref_image()                               #function to actually update the input image in GUI



def display_input_image():                              #function for  updating the input image in the QtGUI
    global img,input_image_copy                         #global variable which will be used in the function
    pics = QPixmap('temp_input.jpg')                    #reads the input image file
    pixmap = pics.scaled(350,350)                       #scales the input file according to the input label size
    inlabel.setPixmap(pixmap)                           #passes the input image to Pixmap
    inlabel.show()                                      #to display the image in label

#function for updating the output image in QTGUI
def display_blur_image():
    global blur_img                                           #global variable which will be used in the function
                                                            #First convert the output image(np array) into qtImage 
    qtimg = QtGui.QImage(blur_img.data,blur_img.shape[1],blur_img.shape[0],blur_img.shape[1],QtGui.QImage.Format_Indexed8)
    pixmap = QtGui.QPixmap.fromImage(qtimg)                 #convert the qimage into a pixmap,
    k_label.setScaledContents(True)                        #scaling the image according to k_label
    k_label.setPixmap(pixmap)                              #load the pixmap in the k_label


def display_ref_image():                                #function for  updating the input image in the QtGUI
    global img,input_image_copy                         #global variable which will be used in the function
    pics = QPixmap('temp_ref_input.jpg')                    #reads the input image file
    pixmap = pics.scaled(350,350)                       #scales the input file according to the input label size
    outlabel.setPixmap(pixmap)                           #passes the input image to Pixmap
    outlabel.show()   

#function for updating the output image in QTGUI
def display_output_image():
    global outimg,in_img                                           #global variable which will be used in the function
                                                            #First convert the output image(np array) into qtImage 
    qtimg = QtGui.QImage(outimg.data,outimg.shape[1],outimg.shape[0],outimg.shape[1],QtGui.QImage.Format_Indexed8)
    pixmap = QtGui.QPixmap.fromImage(qtimg)                 #convert the qimage into a pixmap,
    outlabel.setScaledContents(True)                        #scaling the image according to outlabel
    outlabel.setPixmap(pixmap)                              #load the pixmap in the outlabel




# Ref: https://blogs.mathworks.com/steve/2007/08/13/image-deblurring-introduction/ 
# The concept to understand was w.r.t Full Inverse and Truncated Inverse,i.e. how will the output image look like

def on_deblur_1_button():
    global img,outimg,recent_outimg,blur_img,ref_img                        
    recent_outimg = outimg                                  #preserve the last output
    local_img = img    
    kernel = blur_img
    kernel = np.pad(kernel, [(0, local_img.shape[0] - kernel.shape[0]), (0, local_img.shape[1] - kernel.shape[1])], 'constant')    
    kernel /= kernel.sum()

    ref_r = np.float32(ref_img[:,:,2])    
    max_r = np.max(ref_r)

    fr = np.float32(input_image_copy[:,:,2])
    ref_fr = fr
    fr = np.fft.fft2(fr)
    fr_shift = np.fft.fftshift(fr)

    fg = np.float32(input_image_copy[:,:,1])
    fg = np.fft.fft2(fg)
    fg_shift = np.fft.fftshift(fg)  

    fb = np.float32(input_image_copy[:,:,0])
    fb = np.fft.fft2(fb)
    fb_shift = np.fft.fftshift(fb)

    #FFT for PSF
    h = np.fft.fft2(kernel)
    hshift = np.fft.fftshift(h)

    fh_ishift = np.divide(fr_shift,hshift)
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_r = np.fft.ifft2(fh_ishift)
    img_back_r = np.abs(img_back_r)
    img_back_r_max = np.max(img_back_r)
    #img_back_r = np.divide((img_back_r),np.max(img_back_r))

    fh_ishift = np.divide(fg_shift,hshift)
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_g = np.fft.ifft2(fh_ishift)
    img_back_g = np.abs(img_back_g)
    #img_back_g = np.divide((img_back_g),np.max(img_back_g))

    fh_ishift = np.divide(fb_shift,hshift)
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_b = np.fft.ifft2(fh_ishift)
    img_back_b = np.abs(img_back_b)
    #img_back_b = np.divide((img_back_b),np.max(img_back_b))

    orig_img = cv2.cvtColor(input_image_copy, cv2.COLOR_BGR2RGB)
    sum_img = cv2.merge([img_back_r,img_back_g,img_back_b])
    

    m = mse(input_image_copy, sum_img)
    psnr = 10.0*np.log10((255*255)/m)
    #s= skm.measure.compare_ssim(ref_r.astype("float"),img_back_r)    
    s= skm.measure.compare_ssim(input_image_copy.astype("float"),sum_img,multichannel=True)    


    plt.axis("off")
    plt.suptitle("PSNR: %.3f, MSE: %.3f, SSIM: %.3f" % (psnr,m, s))    
       
    plt.subplot(121),plt.imshow(orig_img)
    plt.title('Input Blurred Image'), plt.xticks([]), plt.yticks([])   
     
    plt.subplot(122),plt.imshow(np.uint8(sum_img))
    plt.title('Output De-blurred Image'), plt.xticks([]), plt.yticks([])  
    '''
    
    plt.subplot(151),plt.imshow(orig_img)
    plt.subplot(152),plt.imshow(img_back_r, cmap = 'gray')
    plt.subplot(153),plt.imshow(img_back_g, cmap = 'gray')
    plt.subplot(154),plt.imshow(img_back_b, cmap = 'gray')
    plt.subplot(155),plt.imshow(sum_img) 
    ''' 

    plt.show()


#Deblur with Truncated Filtering
def on_constrained_deblur():
    global img,outimg,recent_outimg,blur_img,ref_img                        
    recent_outimg = outimg                                  #preserve the last output
    local_img = img    
    kernel = blur_img
    kernel /= kernel.sum()
    kernel = np.float32(np.pad(kernel, [(0, local_img.shape[0] - kernel.shape[0]), (0, local_img.shape[1] - kernel.shape[1])], 'constant')  )
 
    print(kernel.sum())
    R,status = QtGui.QInputDialog.getDouble(wind,"input dialog","enter Radius value",1,1,1e4,2)  #get radius input from the user
    if status is False:                                     #check the status of user input
         print("Entering Radius value cancelled")                #log the false state
         return  
    print(R)

    ref_r = np.float32(ref_img[:,:,2])   
    max_r = np.max(ref_r) 
    ref_r = np.divide(ref_r,np.max(ref_r)) 

    #FFT for Red Channel,Green Channel, Blue Channel

    fr = np.float32(input_image_copy[:,:,2])
    fr_temp = np.divide(fr,np.max(fr))
    fr = np.fft.fft2(fr)
    fr_shift = np.fft.fftshift(fr)

    fg = np.float32(input_image_copy[:,:,1])
    fg = np.fft.fft2(fg)
    fg_shift = np.fft.fftshift(fg)  

    fb = np.float32(input_image_copy[:,:,0])
    fb = np.fft.fft2(fb)
    fb_shift = np.fft.fftshift(fb)

    #FFT for PSF
    h = np.fft.fft2(kernel)
    hshift = np.fft.fftshift(h)

    n,m = fr.shape
    fh_ishift = np.divide(fr_shift,hshift)
    for i in range(0,n):                              
        for j in range(0,m):
            if( ( (i-(n//2))*(i-(n//2)) + (j-(m//2))*(j-(m//2)) ) >= R*R):
                fh_ishift[i,j] = fr_shift[i,j]

    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_r = np.fft.ifft2(fh_ishift)
    img_back_r = np.abs(img_back_r)
    img_back_r_max = np.max(img_back_r)
    #img_back_r = np.divide((img_back_r),np.max(img_back_r))

    #Recovering green channel
    fh_ishift = np.divide(fg_shift,hshift)
    #fh_ishift = fg_shift
    for i in range(0,n):                              
        for j in range(0,m):
            if( ( (i-(n//2))*(i-(n//2)) + (j-(m//2))*(j-(m//2)) ) >= R*R):
                fh_ishift[i,j] = fg_shift[i,j]

    #fh_ishift = fh_ishift + fg_shift
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_g = np.fft.ifft2(fh_ishift)
    img_back_g = np.abs(img_back_g)
    #img_back_g = np.divide((img_back_g),np.max(img_back_g))

    #Recovering blue channel
    fh_ishift = np.divide(fb_shift,hshift)
    #fh_ishift = fb_shift    
    for i in range(0,n):                              
        for j in range(0,m):
            if( ( (i-(n//2))*(i-(n//2)) + (j-(m//2))*(j-(m//2)) ) >= R*R):
                fh_ishift[i,j] = fb_shift[i,j]

    #fh_ishift = fh_ishift + fb_shift
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_b = np.fft.ifft2(fh_ishift)
    img_back_b = np.abs(img_back_b)
    #img_back_b = np.divide((img_back_b),np.max(img_back_b))


    ref_cmp_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    #ref_cmp_img_grey = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)


    orig_img = cv2.cvtColor(input_image_copy, cv2.COLOR_BGR2RGB)
    sum_img = cv2.merge([img_back_r,img_back_g,img_back_b])

    #compare_image1 = (input_image_copy.astype("float"))/np.max(input_image_copy)
    compare_image1 = (ref_cmp_img.astype("float"))/np.max(ref_cmp_img)
    compare_image2 = (sum_img.astype("float"))/np.max(sum_img)

    m = mse(compare_image1,compare_image2)
    psnr = 10.0*np.log10((1*1)/m)
    s= skm.measure.compare_ssim(compare_image1,compare_image2,multichannel=True)


    plt.axis("off")
    plt.suptitle("PSNR: %.3f, MSE: %.3f, SSIM: %.3f" % (psnr,m, s))
       
    plt.subplot(131),plt.imshow(orig_img)
    plt.title('Input Blurred Image'), plt.xticks([]), plt.yticks([])   
     
    plt.subplot(132),plt.imshow(np.uint8(sum_img))
    plt.title('Output De-blurred Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(133),plt.imshow(np.uint8(ref_cmp_img))
    plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])    
    '''
    
    plt.subplot(151),plt.imshow(orig_img)
    plt.title('Input Blurred Image'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(152),plt.imshow(img_back_r, cmap = 'gray')
    plt.title('Deblur_R'), plt.xticks([]), plt.yticks([])     
    plt.subplot(153),plt.imshow(img_back_g, cmap = 'gray')
    plt.title('Deblur_G'), plt.xticks([]), plt.yticks([])     
    plt.subplot(154),plt.imshow(img_back_b, cmap = 'gray')
    plt.title('Deblur_B'), plt.xticks([]), plt.yticks([])     
    plt.subplot(155),plt.imshow(sum_img) 
    plt.title('Deblur_RGB'), plt.xticks([]), plt.yticks([])     
    ''' 

    plt.show()


'''
# This is an equivalent implementation of Truncated Filtering. The results obtained from this method were exactly same as that obtained from previous method.

#Deblur with Kernel 1 button
def on_constrained_deblur():
    global img,outimg,recent_outimg,blur_img                        
    recent_outimg = outimg                                  #preserve the last output
    local_img = img    
    kernel = blur_img
    kernel = np.pad(kernel, [(0, local_img.shape[0] - kernel.shape[0]), (0, local_img.shape[1] - kernel.shape[1])], 'constant')    
    R,status = QtGui.QInputDialog.getDouble(wind,"input dialog","enter Radius value",0,0,1e4,2)  #get gamma input from the user
    if status is False:                                     #check the status of user input
         print("gamma correction cancelled")                #log the false state
         return  
    print(R)
    #FFT for Red Channel,Green Channel, Blue Channel

    fr = np.float32(input_image_copy[:,:,2])
    #fr = np.divide(fr,255)plt.suptitle("PSNR:
    fr = np.fft.fft2(fr)
    fr_shift = np.fft.fftshift(fr)

    fg = np.float32(input_image_copy[:,:,1])
    #fg = np.divide(fg,255)
    #fg = np.fft.fft2(input_image_copy[:,:,1])
    fg = np.fft.fft2(fg)
    fg_shift = np.fft.fftshift(fg)  

    fb = np.float32(input_image_copy[:,:,0])
    #fb = np.divide(fb,255)
    #fb = np.fft.fft2(input_image_copy[:,:,0])plt.suptitle("PSNR:
    fb = np.fft.fft2(fb)
    fb_shift = np.fft.fftshift(fb)

    #FFT for PSF
    h = np.fft.fft2(kernel)
    hshift = np.fft.fftshift(h)

    n,m = fr.shape
    fh_ishift = fr_shift
    for i in range(0,n):                              
        for j in range(0,m):
            if( ( (i-(n/2))*(i-(n/2)) + (j-(m/2))*(j-(m/2)) ) < R*R):
                #fh_ishift[i,j] = 0
                fh_ishift[i,j] = fh_ishift[i,j]/hshift[i,j]

    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_r = np.fft.ifft2(fh_ishift)
    img_back_r = np.abs(img_back_r)
    img_back_r = np.divide((img_back_r),np.max(img_back_r))

    #Recovering green channel
    #fh_ishift = np.divide(fg_shift,hshift)
    fh_ishift = fg_shift
    for i in range(0,n):                              plt.suptitle("PSNR:
        for j in range(0,m):
            if( ( (i-(n/2))*(i-(n/2)) + (j-(m/2))*(j-(m/2)) ) < R*R):
                fh_ishift[i,j] = fh_ishift[i,j]/hshift[i,j]
    #fh_ishift = fh_ishift + fg_shift
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_g = np.fft.ifft2(fh_ishift)
    img_back_g = np.abs(img_back_g)
    img_back_g = np.divide((img_back_g),np.max(img_back_g))

    #Recovering blue channel
    #fh_ishift = np.divide(fb_shift,hshift)
    fh_ishift = fb_shift    
    for i in range(0,n):                              
        for j in range(0,m):
            if( ( (i-(n/2))*(i-(n/2)) + (j-(m/2))*(j-(m/2)) ) < R*R):
                fh_ishift[i,j] = fh_ishift[i,j]/hshift[i,j]
    #fh_ishift = fh_ishift + fb_shift

    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_b = np.fft.ifft2(fh_ishift)plt.suptitle("PSNR:
    img_back_b = np.abs(img_back_b)
    img_back_b = np.divide((img_bacimport cmathk_b),np.max(img_back_b))

    orig_img = cv2.cvtColor(input_image_csudo apt-get install python-skimageopy, cv2.COLOR_BGR2RGB)
    sum_img = cv2.merge([img_back_r,img_back_g,img_back_b])
    
    
    plt.axis("off")
        
    #plt.subplot(121),plt.imshow(orig_img)
    #plt.title('Input Blurred Image'), plt.xticks([]), plt.yticks([])   
     
    #plt.subplot(122),plt.imshow(sum_img)
    #plt.title('Output De-blurred Image'), plt.xticks([]), plt.yticks([])  
   
    
    plt.subplot(151),plt.imshow(orig_img)
    plt.title('Input Blurred Image'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(152),plt.imshow(img_back_r, cmap = 'gray')
    plt.title('Deblur_R'), plt.xticks([]), plt.yticks([])     
    plt.subplot(153),plt.imshow(img_back_g, cmap = 'gray')
    plt.title('Deblur_G'), plt.xticks([]), plt.yticks([])     
    plt.subplot(154),plt.imshow(img_back_b, cmap = 'gray')
    plt.title('Deblur_B'), plt.xticks([]), plt.yticks([])     
    plt.subplot(155),plt.imshow(sum_img) 
    plt.title('Deblur_RGB'), plt.xticks([]), plt.yticks([])     
    

    plt.show()sudo apt-get install python-skimage

'''

def dft_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    exp = np.exp(-6.28J/N )
    dft_m = np.power( exp, i * j ) / N
    return dft_m

#Fourier Transform and Inverse Fourier Transform
#Ref 1: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
#Ref 2: https://stackoverflow.com/questions/19739503/dft-matrix-in-python
# the concept to understand was FFT and IFFT libraries in opencv. and also Matrix method for DFT calculation.


def fourier_transform():
    global img,outimg,recent_outimg                        

    local_img = img
    rows, cols = local_img.shape
    dft_m = dft_matrix(cols)
    dft_of_img = dft_m.dot(local_img).dot(dft_m)
    dft_of_img_shift = np.fft.fftshift(dft_of_img)
    magnitude_spectrum_img = 20*np.log(np.abs(dft_of_img_shift))

    lib_fft = np.fft.ifft2(local_img)
    lib_fft_shift = np.fft.fftshift(lib_fft)
    mag_spectrum_lib = 20*np.log(np.abs(lib_fft_shift))
    
    plt.subplot(131),plt.imshow(local_img, cmap = 'gray')
    plt.title('Input  Image'), plt.xticks([]), plt.yticks([])   
     
    plt.subplot(132),plt.imshow(mag_spectrum_lib, cmap = 'gray')
    plt.title('DFT function output'), plt.xticks([]), plt.yticks([])

    plt.subplot(133),plt.imshow(mag_spectrum_lib, cmap = 'gray')
    plt.title('DFT library output'), plt.xticks([]), plt.yticks([])
    
    plt.show()

    
#This function attempts to de-blur a personal image whose GroundTruth is not known and PSF is not known.
def personal_deblur():
    global img,outimg,recent_outimg,blur_img,input_image_copy,ref_img
                    
    recent_outimg = outimg                                  #preserve the last output
    local_img = in_img                                      #passing global variable into a local variable for operations local to this function
    pxy = np.matrix([[0,1,0], [-1, 4, -1], [0,-1,0]])
    Y_cls,status = QtGui.QInputDialog.getDouble(wind,"input dialog","enter Y value",1,0,1e6,5)  #get gamma input from the user
    if status is False:                                     #check the status of user input
         print("gamma correction cancelled")                #log the false state
         return  
    print(Y_cls)
    kernel = blur_img
    kernel = np.pad(kernel, [(0, local_img.shape[0] - kernel.shape[0]), (0, local_img.shape[1] - kernel.shape[1])], 'constant')
    pxy_pad = np.pad(pxy, [(0, local_img.shape[0] - pxy.shape[0]), (0, local_img.shape[1] - pxy.shape[1])], 'constant') 
    cv2.imwrite('temp_pxy_kernel.png',pxy_pad)

    '''
    ref_r = np.float32(ref_img[:,:,2])    
    ref_r = np.divide(ref_r,np.max(ref_r)) 

    
    ref_g = np.float32(ref_img[:,:,1])    
    ref_g = np.divide(ref_g,np.max(ref_g)) 


    ref_b = np.float32(ref_img[:,:,0])    
    ref_b = np.divide(ref_b,np.max(ref_b)) 
    '''

    #FFT for Red Channel,Green Channel, Blue Channel
    fr = np.float32(input_image_copy[:,:,2])
    fr_temp = np.divide(fr,np.max(fr))
    fr = np.fft.fft2(fr)
    fr_shift = np.fft.fftshift(fr)

    fg = np.float32(input_image_copy[:,:,1])
    fg = np.fft.fft2(fg)
    fg_shift = np.fft.fftshift(fg)  

    fb = np.float32(input_image_copy[:,:,0])
    fb = np.fft.fft2(fb)
    fb_shift = np.fft.fftshift(fb)

    #FFT of PSF and Laplacian Filter
    h = np.fft.fft2(kernel)
    hshift = np.fft.fftshift(h)

    pf = np.fft.fft2(pxy_pad)
    pshift = np.fft.fftshift(pf)
    
    #Computation of Conjugate H and other computations
    hf_conj = np.conjugate(hshift)
    hf_square = np.multiply(np.abs(hshift),np.abs(hshift))
    pf_square = np.multiply(np.abs(pshift),np.abs(pshift))
    pf_temp = np.multiply(pf_square,Y_cls)
    hf_temp =  hf_square + pf_temp
    hf_cls_temp = np.divide(hf_conj,hf_temp)

    #multiplying with input blurred image (frequency domain) for R,G,B
    hf_cls = np.multiply(hf_cls_temp,fr_shift)
    fh_ishift = np.fft.ifftshift(hf_cls)
    img_back_r = np.fft.ifft2(fh_ishift)
    img_back_r = np.abs(img_back_r)
    img_back_r = np.divide((img_back_r),np.max(img_back_r))

    hf_cls = np.multiply(hf_cls_temp,fg_shift)
    fh_ishift = np.fft.ifftshift(hf_cls)
    img_back_g = np.fft.ifft2(fh_ishift)
    img_back_g = np.abs(img_back_g)
    img_back_g = np.divide((img_back_g),np.max(img_back_g))

    hf_cls = np.multiply(hf_cls_temp,fb_shift)
    fh_ishift = np.fft.ifftshift(hf_cls)
    img_back_b = np.fft.ifft2(fh_ishift)
    img_back_b = np.abs(img_back_b)
    img_back_b = np.divide((img_back_b),np.max(img_back_b))

    orig_img = cv2.cvtColor(input_image_copy, cv2.COLOR_BGR2RGB)
    sum_img = cv2.merge([img_back_r,img_back_g,img_back_b])
    
    print(outimg.shape)
    
    plt.axis("off")

    plt.subplot(121),plt.imshow(orig_img)
    plt.title('Input Blurred Image'), plt.xticks([]), plt.yticks([])   
     
    plt.subplot(122),plt.imshow(sum_img)
    plt.title('Output De-blurred Image'), plt.xticks([]), plt.yticks([])       

    '''    
    plt.subplot(151),plt.imshow(orig_img)
    plt.subplot(152),plt.imshow(img_back_r, cmap = 'gray')
    plt.subplot(153),plt.imshow(img_back_g, cmap = 'gray')
    plt.subplot(154),plt.imshow(img_back_b, cmap = 'gray')
    plt.subplot(155),plt.imshow(sum_img) 
    '''

    plt.show()


def image_dft_slow():
    global recent_outimg,outimg,input_image_copy            #global variable which will be used in the function
   
    outimg = input_image_copy[:,:,0]   
    temp = dft(outimg)    
    print(temp.shape)
    plt.imshow(temp)
    plt.show() 

def dft(x):
    local_img = x
    m,n =local_img.shape
    N=m
    y= np.zeros_like(local_img)
    for i in range(0,m):
        print(i)
        for j in range(0,n):
            a = 0
            for k in range(N):
                a += local_img[i,j]*cmath.exp(-2j*cmath.pi*k*n*(1/N))
            y[i,j] = a
    return y

#save the image on to the hard disk, get the filename from QT
def on_save_button():
    global outimg                                           #global variable which will be used in the function
    filename = QtGui.QFileDialog.getSaveFileName()          #get the save filename from file browser of QT
    if filename:
        cv2.imwrite(filename,outimg)                        #if it is valid filename, save the image 
    else:
        print("save image cancelled")                       #otherwise print message and return


#this function performs debluring using Wiener Filter
def weiner_deblur(): 
    global img,outimg,recent_outimg,blur_img,input_image_copy,ref_img                          
    recent_outimg = outimg                                  #preserve the last output
    local_img = in_img                                      #passing global variable into a local variable for operations local to this function
    kernel = blur_img
    #kernel /= kernel.sum()                                  #kernel normalization
    k,status = QtGui.QInputDialog.getDouble(wind,"input dialog","enter K value",1,0,1e6,5)  #get gamma input from the user
    if status is False:                                     #check the status of user input
         print("K value cancelled")                #log the false state
         return  
    print(k)
    y = np.zeros_like(local_img)                             #creating an empty image of same dimensions as that of local image
    kernel = np.pad(kernel, [(0, local_img.shape[0] - kernel.shape[0]), (0, local_img.shape[1] - kernel.shape[1])], 'constant')  
    
    cv2.imwrite('temp_kernel.png',kernel)

    #FFT for Red Channel,Green Channel, Blue Channel

    fr = np.float32(input_image_copy[:,:,2])
    fr = np.fft.fft2(fr)
    fr_shift = np.fft.fftshift(fr)

    fg = np.float32(input_image_copy[:,:,1])
    fg = np.fft.fft2(fg)
    fg_shift = np.fft.fftshift(fg)  

    fb = np.float32(input_image_copy[:,:,0])
    fb = np.fft.fft2(fb)
    fb_shift = np.fft.fftshift(fb)


    #FFT for PSF
    h = np.fft.fft2(kernel)
    hshift = np.fft.fftshift(h)
   
    #Computation for Inverse PSF

    hf_square = np.multiply(np.abs(hshift),np.abs(hshift))
    hf_weiner = np.divide(hf_square,(hf_square + k))
    hf_weiner = np.divide(hf_weiner,np.abs(hshift))

    fh_ishift = np.multiply(fr_shift,hf_weiner)
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_r = np.fft.ifft2(fh_ishift)
    img_back_r = np.abs(img_back_r)
    img_back_r_max = np.max(img_back_r)
    img_back_r = np.divide((img_back_r),np.max(img_back_r))

    fh_ishift = np.multiply(fg_shift,hf_weiner)
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_g = np.fft.ifft2(fh_ishift)
    img_back_g = np.abs(img_back_g)
    img_back_g = np.divide((img_back_g),np.max(img_back_g))

    fh_ishift = np.multiply(fb_shift,hf_weiner)
    fh_ishift = np.fft.ifftshift(fh_ishift)
    img_back_b = np.fft.ifft2(fh_ishift)
    img_back_b = np.abs(img_back_b)
    img_back_b = np.divide((img_back_b),np.max(img_back_b))

    ref_cmp_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.cvtColor(input_image_copy, cv2.COLOR_BGR2RGB)
    sum_img = cv2.merge([img_back_r,img_back_g,img_back_b])

   
    print("Max value")
    print(np.max(ref_cmp_img),np.max(sum_img))

    compare_image1 = (ref_cmp_img.astype("float"))/np.max(ref_cmp_img)
    compare_image2 = (sum_img.astype("float"))/np.max(sum_img)

    print("Max value after normalisation")
    print(np.max(compare_image1),np.max(compare_image2))


    m = mse(compare_image1,compare_image2)
    psnr = 10.0*np.log10((255*255)/m)
    s= skm.measure.compare_ssim(compare_image1,compare_image2,multichannel=True)

    plt.axis("off")
    plt.suptitle("PSNR: %.3f, MSE: %.3f, SSIM: %.3f" % (psnr, m, s))

    plt.subplot(121),plt.imshow(orig_img)

    plt.title('Input Blurred Image'), plt.xticks([]), plt.yticks([])   
     
    #plt.subplot(122),plt.imshow(np.uint8(sum_img))
    plt.subplot(122),plt.imshow(sum_img)
    plt.title('Output De-blurred Image'), plt.xticks([]), plt.yticks([])  

    '''
    plt.subplot(151),plt.imshow(orig_img)
    plt.subplot(152),plt.imshow(img_back_r, cmap = 'gray')
    plt.subplot(153),plt.imshow(img_back_g, cmap = 'gray')
    plt.subplot(154),plt.imshow(img_back_b, cmap = 'gray')
    plt.subplot(155),plt.imshow(sum_img) 
    '''
    plt.show()




#this function performs debluring using Constrained Least Square Error Filter
def cls_deblur(): 
    global img,outimg,recent_outimg,blur_img,input_image_copy,ref_img                        
    recent_outimg = outimg                                  #preserve the last output
    local_img = in_img                                      #passing global variable into a local variable for operations local to this function
    kernel = blur_img
    #kernel /= kernel.sum()                                  #kernel normalization
    pxy = np.matrix([[0,-1,0], [-1, 4, -1], [0,-1,0]])
    Y_cls,status = QtGui.QInputDialog.getDouble(wind,"input dialog","enter Y value",1,0,1e6,5)  #get gamma input from the user
    if status is False:                                     #check the status of user input
         print("Y  cancelled")                              #log the false state
         return  
    print(Y_cls)
    kernel = np.pad(kernel, [(0, local_img.shape[0] - kernel.shape[0]), (0, local_img.shape[1] - kernel.shape[1])], 'constant')
    pxy_pad = np.pad(pxy, [(0, local_img.shape[0] - pxy.shape[0]), (0, local_img.shape[1] - pxy.shape[1])], 'constant') 
    cv2.imwrite('temp_pxy_kernel.png',pxy_pad)

    #FFT for Red Channel,Green Channel, Blue Channel
    fr = np.float32(input_image_copy[:,:,2])
    fr = np.fft.fft2(fr)
    fr_shift = np.fft.fftshift(fr)

    
    fg = np.float32(input_image_copy[:,:,1])
    fg = np.fft.fft2(fg)
    fg_shift = np.fft.fftshift(fg)  

    fb = np.float32(input_image_copy[:,:,0])
    fb = np.fft.fft2(fb)
    fb_shift = np.fft.fftshift(fb)

    #FFT of PSF and Laplacian Filter
    h = np.fft.fft2(kernel)
    hshift = np.fft.fftshift(h)

    pf = np.fft.fft2(pxy_pad)
    pshift = np.fft.fftshift(pf)
    
    #Computation of Conjugate H and other computations
    hf_conj = np.conjugate(hshift)
    hf_square = np.multiply(np.abs(hshift),np.abs(hshift))
    pf_square = np.multiply(np.abs(pshift),np.abs(pshift))
    pf_temp = np.multiply(pf_square,Y_cls)
    hf_temp =  hf_square + pf_temp
    hf_cls_temp = np.divide(hf_conj,hf_temp)

    #multiplying with input blurred image (frequency domain) for R,G,B
    hf_cls = np.multiply(hf_cls_temp,fr_shift)
    fh_ishift = np.fft.ifftshift(hf_cls)
    img_back_r = np.fft.ifft2(fh_ishift)
    img_back_r = np.abs(img_back_r)
    img_back_r_max = np.max(img_back_r)
    img_back_r = np.divide((img_back_r),np.max(img_back_r))

    hf_cls = np.multiply(hf_cls_temp,fg_shift)
    fh_ishift = np.fft.ifftshift(hf_cls)
    img_back_g = np.fft.ifft2(fh_ishift)
    img_back_g = np.abs(img_back_g)
    img_back_g = np.divide((img_back_g),np.max(img_back_g))

    hf_cls = np.multiply(hf_cls_temp,fb_shift)
    fh_ishift = np.fft.ifftshift(hf_cls)
    img_back_b = np.fft.ifft2(fh_ishift)
    img_back_b = np.abs(img_back_b)
    img_back_b = np.divide((img_back_b),np.max(img_back_b))

    ref_cmp_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.cvtColor(input_image_copy, cv2.COLOR_BGR2RGB)
    sum_img = cv2.merge([img_back_r,img_back_g,img_back_b])
    
    compare_image1 = (ref_cmp_img.astype("float"))/np.max(ref_cmp_img)
    compare_image2 = (sum_img.astype("float"))/np.max(sum_img)

    m = mse(compare_image1,compare_image2)
    psnr = 10.0*np.log10((255*255)/m)
    s= skm.measure.compare_ssim(compare_image1,compare_image2,multichannel=True)

    
    plt.axis("off")
    plt.suptitle("PSNR: %.3f, MSE: %.3f, SSIM: %.3f" % (psnr, m, s))

    plt.subplot(121),plt.imshow(orig_img)
    plt.title('Input Blurred Image'), plt.xticks([]), plt.yticks([])   
     
    #plt.subplot(122),plt.imshow(np.uint8(sum_img))
    plt.subplot(122),plt.imshow(sum_img)
    plt.title('Output De-blurred Image'), plt.xticks([]), plt.yticks([])       

    '''    
    plt.subplot(151),plt.imshow(orig_img)
    plt.subplot(152),plt.imshow(img_back_r, cmap = 'gray')
    plt.subplot(153),plt.imshow(img_back_g, cmap = 'gray')
    plt.subplot(154),plt.imshow(img_back_b, cmap = 'gray')
    plt.subplot(155),plt.imshow(sum_img) 
    '''

    plt.show()



# Reference : https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err/= float(imageA.shape[0] * imageA.shape[1])
    '''
    #The below statement perform the same function, but in loop
    sum_j = 0
    sum_i = 0
    for i in range(0,imageA.shape[0]):                              
        for j in range(0,imageA.shape[1]):
            err2 = imageA[i,j]-imageB[i,j]
            sum_j = sum_j + err2*err2   
    err2 = sum_j/(imageA.shape[0]*imageA.shape[1])
    print("MSE 1 and MSE 2")
    print(err, err2)
    '''
    return err



#This function blurs the input image 
#https://stackoverflow.com/questions/35192550/wiener-filter-for-image-deblur : for making kernel size same as image size
def image_blur(): 
    global outimg,recent_outimg,in_img                      #global variable which will be used in the function
    recent_outimg = outimg                                  #preserve the last output
    local_img = in_img                                      #passing global variable into a local variable for operations local to this function
    y= np.zeros_like(local_img)                             #creating an empty image of same dimensions as that of local image
    ks=5                                                   #convert the input into int
    ws=2*ks+1                                               #creating odd number based on input 
    kernel = np.ones((ws,ws),np.float32)/(ws*ws)
    kernel = np.pad(kernel, [(0, local_img.shape[0] - kernel.shape[0]), (0, local_img.shape[1] - kernel.shape[1])], 'constant')    
    

    #print(kernel.shape)

    f = np.fft.fft2(local_img)
    fshift = np.fft.fftshift(f)
    f_magnitude_spectrum = 20*np.log(np.abs(fshift))    

    h = np.fft.fft2(kernel)
    hshift = np.fft.fftshift(h)
    h_magnitude_spectrum = 20*np.log(np.abs(hshift))    

    #print(hshift.shape)

    fh_ishift = np.fft.ifftshift((fshift * hshift))
    img_blur = np.fft.ifft2(fh_ishift)
    img_blur = np.abs(img_blur)

    #recover image from blurred image

    img_blur_f = np.fft.fft2(img_blur)
    img_blur_f_shift = np.fft.fftshift(img_blur_f)
    img_blur_magnitude_spectrum = 20*np.log(np.abs(img_blur_f_shift))  

    img_back_ishift = np.fft.ifftshift((img_blur_f_shift / hshift))
    img_back = np.fft.ifft2(img_back_ishift)
    img_back = np.abs(img_back)*255


    plt.subplot(221),plt.imshow(local_img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(h_magnitude_spectrum, cmap = 'gray')
    plt.title('PSF'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(img_blur, cmap = 'gray')
    plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(img_back, cmap = 'gray')
    plt.title('Deblurred Image'), plt.xticks([]), plt.yticks([])


    plt.show()




#below are the different buttons for the QT UI, 
#these have been created manually and so they need to be alligned manually


#Load Reference Image (Unblurred)

input_ref_img = QtGui.QPushButton("Load Ref. Image",wind)      #Title of the Button
input_ref_img.clicked.connect(load_ref_image)               #Button Action on Click (it will call the function)
input_ref_img.resize(250,30)                                  #Dimensions of the Button 
input_ref_img.move(40,440)                                    #Position of the Button 

#load button : Loads input image

ldbtn = QtGui.QPushButton("Load Blur Image",wind)                #Title of the Button
ldbtn.clicked.connect(on_load_button)                       #Button Action on Click (it will call the function)
ldbtn.resize(250,30)                                  #Dimensions of the Button 
ldbtn.move(40,480)                                    #Position of the Button                                        

#Load Blur Kernel

input_hist = QtGui.QPushButton("Load Blur Kernel",wind)      #Title of the Button
input_hist.clicked.connect(load_blur_kernel)               #Button Action on Click (it will call the function)
input_hist.resize(250,30)                                  #Dimensions of the Button 
input_hist.move(40,520)                                    #Position of the Button 




#Full Inverse

eq_hist_btn = QtGui.QPushButton("Full Inverse Deblur",wind)  #Title of the Button     
eq_hist_btn.clicked.connect(on_deblur_1_button)             #Button Action on Click (it will call the function)
eq_hist_btn.resize(250,30)                                #Dimensions of the Button 
eq_hist_btn.move(450,450)                                 #Position of the Button  



#Quit button (to exit the application)

btn = QtGui.QPushButton("Quit",wind)                        #Title of the Button
btn.clicked.connect(QtCore.QCoreApplication.instance().quit)#Button Action on Click (it will call the function)
btn.resize(250,30)                                          #Dimensions of the Button 
btn.move(40,560)                                            #Position of the Button 


#DFT computation

in_hist_btn = QtGui.QPushButton("DFT Computation",wind)       #Title of the Button
in_hist_btn.clicked.connect(fourier_transform)#(log_transformation)             #Button Action on Click (it will call the function)
in_hist_btn.resize(250,30)                                  #Dimensions of the Button 
in_hist_btn.move(900,400)                                   #Position of the Button 



#Box Blur in Frequency Domain

gblur_btn = QtGui.QPushButton("Image Blur",wind)            #Title of the Button
gblur_btn.clicked.connect(image_blur)                       #Button Action on Click (it will call the function)
gblur_btn.resize(250,30)                                    #Dimensions of the Button 
gblur_btn.move(900,440)                                     #Position of the Button 


#Wiener Filter

gamma_btn = QtGui.QPushButton("Wiener Filter",wind)          #Title of the Button
gamma_btn.clicked.connect(weiner_deblur)                   #Button Action on Click (it will call the function)
gamma_btn.resize(250,30)                                    #Dimensions of the Button 
gamma_btn.move(900,480)                                     #Position of the Button 


#CLS Deblur

btn = QtGui.QPushButton("CLS Deblur",wind)        #Title of the Button
btn.clicked.connect(cls_deblur)                   #Button Action on Click (it will call the function)
btn.resize(250,30)                                          #Dimensions of the Button 
btn.move(900,520)                                           #Position of the Button 


#Save button

save_btn = QtGui.QPushButton("Save",wind)                   #Title of the Button
save_btn.clicked.connect(on_save_button)                    #Button Action on Click (it will call the function)
save_btn.resize(250,30)                                     #Dimensions of the Button 
save_btn.move(900,560)                                      #Position of the Button 



#Truncated Inverse

undo_last_btn = QtGui.QPushButton("Truncated Inverse",wind)         #Title of the Button
undo_last_btn.clicked.connect(on_constrained_deblur)          #Button Action on Click (it will call the function)
undo_last_btn.resize(250,30)                                 #Dimensions of the Button 
undo_last_btn.move(450,500)                                  #Position of the Button 



#Personal Pic Deblur button

undo_all_btn = QtGui.QPushButton("PP_deblur",wind)  #Title of the Button
undo_all_btn.clicked.connect(personal_deblur)             #Button Action on Click (it will call the function)
undo_all_btn.resize(250,30)                                 #Dimensions of the Button 
undo_all_btn.move(450,550)                                  #Position of the Button 

wind.show()                                                 #Show the widget (run the UI application)




sys.exit(app.exec_())                                       #exit the application
