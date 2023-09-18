#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

# In[2]:

from sympy import primenu
#danh sách tên đối tượng tương ứng với nhãn dán lable
subjects = ["", "Ramiz Raja", "Elvis Presley","Lap"]

#hàm phát hiện khuân mặt sử dụng opencv
def detect_face(img):
    #chuyển ảnh về màu xám vì opencv sử dụng hình ảnh màu xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Tải trình phát hiện khuôn mặt bằng LBP sử dụng classcv2.CascadeClassifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #result is a list of faces
    #phát hiện khuân mặt bằng detectMultiScale phát hiện ảnh đa tỉ lệ
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #nếu không có khuôn mặt trả về none
    if (len(faces) == 0):
        return None, None

    #chỉ lấy face đầu tiên
    (x, y, w, h) = faces[0]

    #trả về vị trí của khuân mặt đầu tiền (faces[0])
    return gray[y:y+w, x:x+h], faces[0]

#hàm đọc ảnh và trả về 2 mảng faces và lables(có cùng kích thước) là số khuôn mặt và số nhãn lable cho từng khuôn mặt
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #lấy tên các thư mục có trong data_folder_path
    dirs = os.listdir(data_folder_path)
    
    #list chứa tất cả faces
    faces = []
    #list chứa tất cả các nhãn
    labels = []
    
    #duyệt từng thư mục và đọc các hình ảnh bên trong
    for dir_name in dirs:
        
        #tất cả thư mục đề bắt đầu bằng 's' nên hãy bỏ qua những thư mục không liên quan
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #bắt đầu trích xuất nhãn từ tên thư mục
        #tên thư mục các dạng là slable
        #xóa kí tự s để lấy được nhãn lable
        label = int(dir_name.replace("s", ""))
        
        #tạo đường dẫn thư mục chứa hình ảnh
        #ví dụ subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #lấy tên hình ảnh trong các thư mục chủ đề đã tạo đường dẫn ở trên subject_dir_path
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #đọc từ hình ảnh
        #sau đó phát hiện khuân mặt có trong ảnh và thêm khuôn mặt vào list khuôn mặt (faces)
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #tương tự như trên subject_dir_path xây dựng đường dẫn hình ảnh
            #ví dụ image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #đọc ảnh
            image = cv2.imread(image_path)
            
            #hiển thị hình ảnh ra
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #hàm detect_face sử dụng để phát hiện khuôn mặt
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #nếu phát hiện khuôn mặt
            #thêm khuôn mặt và nhãn vào faces, lables
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

#chuẩn bị dữ liệu
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
print("np arrr",np.array(labels))
print("np arrr",labels)

#tạo bộ nhận dạng khuôn mặt LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#tạo bộ nhận dạng khuôn mặt EigenFaceRecognizer
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#tạo bộ nhận dạng khuôn mặt FisherFaceRecognizer
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

#bắt đầu train dữ liệu đã chuẩn bị ở trên
face_recognizer.train(faces, np.array(labels))

# hàm vẽ hình chữ nhật xung quanh khuôn mặt
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#hàm viết chữ lên ảnh với tạo độ x, y
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

#hàm thực hiện nhận diện
def predict(test_img):
    #tạo một bản sao của hình ảnh
    img = test_img.copy()
    #sử dụng hàm detect face đã viết ở trên để phát hiện khuôn mặt
    face, rect = detect_face(img)
    if face is None:
        return img, None
    #dự đoán hình ảnh bằng cách sử dụng tính nhận diện khuôn mặt cuả bộ nhận diện khuôn mặt face_recognizer đã tạo ở trên
    #trả về lable và confidence
    label, confidence = face_recognizer.predict(face)
    #lấy tên của lable
    print(confidence)
    if confidence>40:
        return img, None
    label_text = subjects[label]
    print(confidence)

    #vẽ hình xung quanh khuôn mặt được nhận diện
    draw_rectangle(img, rect)
    #vẽ tên được dự đoán
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img, label

#bắt đầu quá trình nhận diện bằng 2 ảnh test

print("Predicting images...")

#load ảnh test
# test_img1 = cv2.imread("test-data/test1.jpg")
# test_img2 = cv2.imread("test-data/test2.jpg")
# test_img3 = cv2.imread("test-data/test3.jpg")
# #thực hiện dự đoán
# predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)
# predicted_img3 = predict(test_img3)
# print("Prediction complete")
#
# #hiển thị ảnh
# cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
# cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
# cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400,500)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# cv2.destroyAllWindows()
# cv2.waitKey(2)
# cv2.destroyAllWindows()
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    predicted_img2,label = predict(frame)
    print(label)
    if not ret:
        print("Error: failed to capture image")
        break

    cv2.imshow('OpenCV', predicted_img2 )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

