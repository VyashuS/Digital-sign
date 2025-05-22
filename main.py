import cv2
import numpy as np


sign = cv2.imread('signature.jpeg',cv2.IMREAD_COLOR)
sign = cv2.resize(sign,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
print('shape',sign.shape)
cv2.imshow('sign',sign)
cv2.waitKey(0)
cv2.destroyAllWindows()

sign_crop = sign[20:100,5:220]
cv2.imshow('sign',sign_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
sign_gray = cv2.cvtColor(sign_crop,cv2.COLOR_BGR2GRAY)
alpha_sign = cv2.adaptiveThreshold(sign_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 17)
cv2.imshow('alpha sign',alpha_sign)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
blue = sign_crop.copy()
blue[:,:,:] = [255,0,0]
cv2.imshow('blue mask',blue)
cv2.waitKey(0)

blended_sign = cv2.addWeighted(sign_crop,1,blue,0.8,0)
cv2.imshow('coloured sign',blended_sign)
cv2.waitKey(0)

b,g,r = cv2.split(blended_sign)

new = [b,g,r,alpha_sign]

digital_sign = cv2.merge(new,4)
digital_sign2 = cv2.bitwise_or(digital_sign,digital_sign,mask=alpha_sign)
cv2.imwrite('newalpha_sign.png',digital_sign2)
cv2.imwrite('new_sign.png',digital_sign)
cv2.imshow('final result',digital_sign)
cv2.waitKey(0)
cv2.destroyAllWindows()
