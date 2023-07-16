import cv2

#storing the image in a variable so that we can change it at any time
img_file='images.jpg'

#classifier file stores the pre_trained classifier
classifier_file='cars.xml'

#cv2 reads the img_file and stores it in img
img=cv2.imread(img_file)

#convert the image to greyscales
black_white=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cascade classifier is used to classify the image based on the xml file, that whether it qualifies to be a car or not
car_tracker=cv2.CascadeClassifier(classifier_file)

#detect cars on multi scales throughout the picture, 'cars' stores the coordinates of the cars detected
cars=car_tracker.detectMultiScale(black_white)

for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('image window',img)
cv2.waitKey()