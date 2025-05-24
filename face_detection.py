import cv2

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

current_face_number = 0
total_faces_detected = 0

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    new_faces = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            new_faces += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    if new_faces > current_face_number:
        total_faces_detected += new_faces - current_face_number
        print("Facce rilevate: " + str(total_faces_detected))
        
    current_face_number = new_faces

    cv2.imshow('Face Detection', frame)

cap.release()
cv2.destroyAllWindows()
