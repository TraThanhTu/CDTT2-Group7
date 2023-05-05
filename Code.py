import cv2
import dlib
import math
import time
import numpy as np

# Phát hiện xe
car_detect = cv2.CascadeClassifier('car_detect_harrcascade.xml')
video =cv2.VideoCapture('highway.mp4')

# Định nghĩa tham số dài và rộng
f_width = 1280
f_height = 720
# Cài đặt tham số: số điểm ảnh/met
pixels_per_meter = 1
# Các tham số phục vụ việc theo dõi
frame_idx= 0
car_number= 0
fps= 0

carTracker ={}
carNumber = {}
carStartPosition = {}
carCurentPosition= {}
speed= [None]*1000

# Hàm xóa các Tracker không tốt
def remove_bad_tracker():
    global  carTracker, carStartPosition, carCurentPosition
    # Xóa các car tracker không tốt
    delete_id_list = []

    #Duyệt qua các xe
    for car_id in carTracker.keys():
        #với các xe mà cái xác nhận theo dõi <4 thì đưa vào danh sách xóa
        if carTracker[car_id].update(image) < 4:
            delete_id_list.append(car_id)
    #Thực hiện xóa xe
    for car_id in delete_id_list:
        carTracker.pop(car_id, None)
        carStartPosition.pop(car_id, None)
        carCurentPosition.pop(car_id, None)
    return
#Hàm tính toán tốc độ
def calculate_speed(startPosition, curentPosition, fbs):
    global pixels_per_meter

    #tinh toán khoảng cách di chuyên pixel
    distance_in_pixels= math.sqrt(math.pow(curentPosition[0]-startPosition[0],2)+math.pow(curentPosition[1]-startPosition[1],2))

    #tính toán khoảng cách di chuyển bằng met
    distance_in_meters = distance_in_pixels / pixels_per_meter

    #Tính tốc m/s
    speed_in_meters_per_second = distance_in_meters *fps
    #Quy đổi sang km/h
    speed_in_km_per_hour = speed_in_meters_per_second*3.6
    return speed_in_km_per_hour

while True:
    start_time = time.time()
    _, image = video.read()
    if image is None:
        break
    image = cv2.resize(image,(f_width, f_height))
    output_image = image.copy()
    frame_idx += 1
    remove_bad_tracker()
    #Thực hiện nhận diễn mỗi 10 frame
    if not(frame_idx %10):
        #thực hiện detect xe trong hình
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Chuyển sang thang màu xám
        cars = car_detect.detectMultiScale(gray, 1.2, 13,18, (24, 24)) # Phát hiện nhiều xe cùng lúc

        #Duyệt qua các xe được phát hiện
        for (_x, _y, _w, _h) in cars:
            x= int(_x)
            y= int(_y)
            w= int(_w)
            h= int(_h)

            #Tính tâm của xe
            x_center= x+ 0.5*w
            y_center= y+ 0.5*h

            matchCarID= None
            #Duyệt các xe được theo dõi
            for carID in carTracker.keys():
                #Lay vi tri cua cac xe da theo doi
                trackedPosition= carTracker[carID].get_position()
                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())
                #tinh tam diem cua xe đã theo dõi
                t_x_center = t_x +0.5*t_w
                t_y_center = t_y +0.5*t_h
                #kiem tra xem xe da duoc theo doi hay chuaw
                if( t_x <= x_center<=(t_x+t_w)) and (t_y<= y_center<=(t_y+t_h)) and (x<= t_x_center <=(x+w)) and (y<= t_y_center <= (y+h)):
                    matchCarID = carID

            #neu khong phai xe ca duoc theo doi thi tao doi tuong theo doi moi, creat a new tracking object
            if matchCarID is None:

                tracker =dlib.correlation_tracker()
                tracker.start_track(image, dlib.rectangle(x, y, x+w, y+h))

                carTracker[car_number]= tracker
                carStartPosition[car_number]=[x,y,w,h]

                car_number +=1
    #thuc hien update vi tri cua cac xe
    for carID in carTracker.keys():
        trackedPosition = carTracker[carID].get_position()

        t_x = int(trackedPosition.left())
        t_y = int(trackedPosition.top())
        t_w = int(trackedPosition.width())
        t_h = int(trackedPosition.height())

        cv2.rectangle(output_image, (t_x, t_y),(t_x+t_w, t_y+t_h), (250,0,0), 4) #tạo hình chữ nhật xung quanh
        carCurentPosition[carID]=[t_x, t_y, t_w, t_h]

    #tinh toan frame moi giay
    end_time= time.time()
    if not(end_time== start_time):
        fps =1.0/(end_time - start_time)

    # lặp qua các xe đã được theo dõi và tính tốc độ
    for i in carStartPosition.keys():
        [x1, y1, w1, h1] = carStartPosition[i]
        [x2, y2, w2, h2] = carCurentPosition[i]

        carStartPosition[i]= [x2,y2,w2,h2]
        #Neu co xe di chuyen thiii
        if[x1,y1,w1,h1]!= [x2,y2,w2,h2]:
            #Nếu như chưa tính toán tốc độ và tọa độ hiện tại <200 thì tính toán tốc độ
            if(speed[i] is None or speed[i]==0) and y2< 200:
                speed[i]= calculate_speed([x1,y1,w1,h1],[x2,y2,w2,h2], fps)

            #Nếu như đã tính tốc do va xe vuo qua tung do 200 thi hien thi tong do
            if speed[i] is not None and y2 >= 200:
                cv2.putText(output_image, str(int(speed[i])) + "km/h",
                            (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,225,225),2)
    cv2.putText(output_image, "VEHICLE COUNTER : " + str(car_number), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow('video', output_image)
    #detect phim Q
    if cv2.waitKey(13)==ord('q'):
        break

cv2.destroyAllWindowns()
cap.release()








