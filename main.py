import cv2 
from gui_buttons import Buttons

#OpenCV DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights",
                      "dnn_model/yolov4-tiny.cfg")    # Import model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), 
                     scale=1/255.0)

#Load class names
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
        
#Initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#Initialize buttons
button_classes = Buttons()
button_classes.add_button("Person", 10, 10)
#put the botton bellow the previous one
button_classes.add_button("Car", 10, 70)


def click_event(event, x, y, flags, params):
    global button_classes
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at: ", x, y)
        button_classes.button_click(x, y)
                
#Create a window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_event)
 
while True:
    #Get frame
    ret, frame = cap.read()
    #Get active buttons list
    active_buttons = button_classes.active_buttons_list() 
    print("Active buttons: ", active_buttons)
    
    #Object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_ids, scores, bbox in zip(class_ids, 
                                       scores, 
                                       bboxes):
        (x, y, w, h) = bbox
        
        #Load class name
        class_name = classes[class_ids]
        
        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y -10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), ((x + w, y + h)),
                          (0, 255, 0), 2)
        
        #Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #Draw label
        cv2.putText(frame, class_name, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    
    #Button to close the window
    cv2.rectangle(frame, (frame.shape[1] - 160, 0), 
                  (frame.shape[1], 50), 
                  (0, 0, 0), -1)
    cv2.putText(frame, "Press 'q' to close", (frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    #Display buttons
    button_classes.display_buttons(frame)
    #Show frame
    cv2.imshow("Frame", frame)
    #Quit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()