import cv2
import face_recognition as fr
import os
import numpy
import pyttsx3
from datetime import datetime

id1 = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0'


def talk(msg):
    # turn on pyttsx3 engine
    engine = pyttsx3.init()
    engine.setProperty('voice', id1)

    # rate
    engine.setProperty('rate', 160)  # setting up new voice rate

    # pronounce message
    engine.say(msg)
    engine.runAndWait()


# create database
path = 'Employees'
my_pictures = []
employees_names = []
employees_list = os.listdir(path)

for name in employees_list:
    img = cv2.imread(f'{path}/{name}')
    my_pictures.append(img)
    employees_names.append(os.path.splitext(name)[0])


# encode images
def encode(pics):
    coded_list = []

    # img to rgb
    for pic in pics:
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(pic)[0]
        coded_list.append(encode)

    return coded_list


# record income
def income_report(person):
    file = open('register.csv', 'r+')
    data_list = file.readlines()
    register_names = []

    for line in data_list:
        income = line.split(',')
        register_names.append(income[0])
    if person not in register_names:
        time = datetime.now()
        time_now = time.strftime('%H:%M:%S')
        file.writelines(f'\n{person}, {time_now}')


print(employees_names)
coded_employees_list = encode(my_pictures)

# take a webcam image
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# read the camera image
success, image = capture.read()
if not success:
    print('The capture could not be taken.')
else:
    # recognize face in capture
    face_capture = fr.face_locations(image)

    coded_face_capture = fr.face_encodings(image, face_capture)

    # find matches
    for coded_face, face_place in zip(coded_face_capture, face_capture):
        match = fr.compare_faces(coded_employees_list, coded_face)
        distances = fr.face_distance(coded_employees_list, coded_face)

        print(distances)
        match_index = numpy.argmin(distances)

        # show matches
        if distances[match_index] > 0.6:
            talk('No coincide con ninguno de nuestros empleados')
        else:
            # Find the name of the employee
            name = employees_names[match_index]
            talk(f'Welcome, {name.split(" ")[0]}.')

            y1, x2, y2, x1 = face_place
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, employees_names[match_index].split(' ')[0], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (255, 255, 255), 2)

            income_report(name)

            # show the image obtained
            cv2.imshow('Capture', image)

            # keep window open
            cv2.waitKey(0)
