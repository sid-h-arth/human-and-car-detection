import cv2

#using the webcam for realtime vid
img_file='images.jpg'
video=cv2.VideoCapture(0)

#stores the info in the xml file
car_tracker_file='cars.xml'

#filters the info through the classifier based on the xml file
car_tracker=cv2.CascadeClassifier(car_tracker_file)

human_tracker_file='pedestrains.xml'

human_tracker=cv2.CascadeClassifier(human_tracker_file)

while True:
    #reading frame by frame from the video
    read_successfull,frame=video.read()

    #increase speed
    if read_successfull:
        gray_scaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    cars=car_tracker.detectMultiScale(gray_scaled_frame)
    human=human_tracker.detectMultiScale(gray_scaled_frame)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    for (x,y,w,h) in human:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow('image window',frame)
    key=cv2.waitKey(1)
    
    if key==81 or key==113:
        break
video.release()

