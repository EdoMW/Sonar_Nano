
# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read and display each frame
    ret, img = cap.read()
    cv2.imshow('a', img)
    k = cv2.waitKey(125)
    # Specify the countdown
    j = 30
    # set the key for the countdown to begin
    if k == ord('q'):
        while j >= 10:
            ret, img = cap.read()
            # Display the countdown after 10 frames so that it is easily visible otherwise,
            # it will be fast. You can set it to anything or remove this condition and put
            # countdown on each frame
            if j % 10 == 0:
                # specify the font and draw the countdown using puttext
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(j // 10), (250, 250), font, 7, (255, 255, 255), 10, cv2.LINE_AA)
            cv2.imshow('a', img)
            cv2.waitKey(125)
            j = j - 1
        else:
            ret, img = cap.read()
            # Display the clicked frame for 1 sec.
            # You can increase time in waitKey also
            cv2.imshow('a', img)
            cv2.waitKey(1000)
            # Save the frame
            cv2.imwrite('D:/downloads/camera.jpg', img)
    # Press Esc to exit
    elif k == 27:
        break
cap.release()
cv2.destroyAllWindows()
