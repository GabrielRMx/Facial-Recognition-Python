import cv2
import face_recognition as fr

# load img

base_pic = fr.load_image_file('Employees/your_img.jpg')
test_pic = fr.load_image_file('Employees/Brad Pitt.jpg')

# img to rgb
base_pic = cv2.cvtColor(base_pic, cv2.COLOR_BGR2RGB)
test_pic = cv2.cvtColor(test_pic, cv2.COLOR_BGR2RGB)

# face tracking
face_placing = fr.face_locations(base_pic)[0]
coded_face_1 = fr.face_encodings(base_pic)[0]

# face tracking test
face_placing_test = fr.face_locations(test_pic)[0]
coded_face_2 = fr.face_encodings(test_pic)[0]

# show rectangle
cv2.rectangle(base_pic,
              (face_placing[3], face_placing[0]),
              (face_placing[1], face_placing[2]),
              (0, 255, 0),
              2)

cv2.rectangle(test_pic,
              (face_placing_test[3], face_placing_test[0]),
              (face_placing_test[1], face_placing_test[2]),
              (0, 255, 0),
              2)

# comparison
results = fr.compare_faces([coded_face_1], coded_face_2)

# measure distance
distance = fr.face_distance([coded_face_1], coded_face_2)

# show results
cv2.putText(test_pic,
            f'{results[0]} {distance.round(2)[0]}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2)

# show pictures
cv2.imshow('base pic', base_pic)
cv2.imshow('test pic', test_pic)

# keep the program open
cv2.waitKey(0)
