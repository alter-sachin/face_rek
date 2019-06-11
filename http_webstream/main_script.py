import face_recognition
import cv2
import numpy as np
import os
from threading import Thread
import pickle
from websocket import create_connection
import time

import cv2
import imagezmq
from queue import Queue

from queue import *
import collections
from datetime import timedelta

q_count = 0
detect_q = collections.deque(maxlen=2)
detect_time = collections.deque(maxlen=2)

people_dict = {}


count = 0
WS = 'ws://118.185.61.235:8090'
now_frame = ''
ws = ''


def on_message(message):
    print(len(str(message)))
    # pipe.stdin.write(message)


def on_error(error):
    print(error)


def on_close():
    print("### closed ###")


def on_open():
    def run(*args):
        for i in range(3):
            time.sleep(1)
            ws.send("Hello %d" % i)
        time.sleep(1)
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())


def checkLock(lst):
    if len(detect_q) == 2:
        ele = lst[0]
        chk = True
        for item in lst:
            if ele != item:
                chk = False
                break
        if (chk == True):
            print("Equal")
            return True
        else:
            print("Not Equal")
            return False
    else:
        return False
        # Comparing each element with first item  .....if one at a time....


def threadFrameGet(threadname, q):
    # Initialize some variables
    # create_connection(WS)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    global count
    global detect_time
    timefrompi_frame = []
    while True:
        # Grab a single frame of video
        count = count + 1
        cap = cv2.VideoCapture('http://118.185.61.236:8000/html/cam_pic.php')
        rpi_name = 0

        #difference = timeDelta(detect_time[0],detect_time[1])
        # print(difference)
        # print(count)
        ret, frame = cap.read()
        #rpi_name, image = image_hub.recv_image()
        #rpi_name, jpg_buffer = image_hub.recv_jpg()
        #image = cv2.imdecode(np.fromstring(jpg_buffer, dtype='uint8'), -1)
        frame_sent_from_pi_at = rpi_name
        frame_received_at = time.time()
        time_to_receive_frame = frame_received_at - frame_sent_from_pi_at
        print("time taken to receive framefrom pi"+str(time_to_receive_frame))
       # image_hub.send_reply(b'OK')
        # cv2.imshow(rpi_name, image) # 1 window for each RPi
        # cv2.waitKey(1)
        #img = cv2.imread(image,0)
        image = frame

        timefrompi_frame.append([rpi_name, image])
        # print(image)
        #global now_frame
        #now_frame = image
        #frame = timefrompi_frame[-1][1]
        # cv2.imwrite("hello.png",image)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        process_this_frame = True
        #print("printing frame")
        # print(rgb_small_frame)
        # Only process every other frame of video to save time
        if process_this_frame:
            print("pushing frame")
            q.put([timefrompi_frame[-1][0], rgb_small_frame])
            # time.sleep(1/10000)
        #    person_name = recognise_person(rgb_small_frame)
        #    print(person_name)
            # call the other function here..... that contains the below lines.
        process_this_frame = not process_this_frame
        # return rgb_small_frame
    # except KeyboardInterrupt:
    #    print("GoodBye")
    # finally:
        # Release handle to the webcam
    #    video_capture.release()
    #    cv2.destroyAllWindows()


def recognise_person(threadname, q):
    # access names that are keys
    #global now_frame
    global q_count
    global detect_q
    global people_dict
    unlock_time = 0
    initial_face_names = list(all_face_encodings.keys())
    for name_keys in initial_face_names:
        # initialise all values of names as 0, which means no one has been detected.
        people_dict[str(name_keys)] = 0

    print(people_dict)
    while True:
        print("inside REKO")
        timefrompi2_frame = q.get()
        frame = timefrompi2_frame[1]
        time_of_frame = timefrompi2_frame[0]
        print(time_of_frame)

        if frame is None:
            continue
        # print(frame)
        rgb_small_frame = frame
        total_face_names = list(all_face_encodings.keys())
        # access values of encodings as a numpy array.
        total_face_encodings = np.array(list(all_face_encodings.values()))
        # Find all the faces and face encodings in the current frame of video
        print("starting to find location of person in image")
        before_locations = time.time()
        face_locations = face_recognition.face_locations(
            rgb_small_frame, number_of_times_to_upsample=1, model="hog")
        print(face_locations)
        after_locations = time.time()
        difference_locations = after_locations - before_locations
        print("time to find person location is"+ str(difference_locations))

        before_encoding = time.time()
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        after_encoding = time.time()
        time_to_encode = after_encoding - before_encoding
        print("time taken to encode is"+str(time_to_encode))
        # print(face_encodings)
        #global count
        #count = count + 1
        # print(count)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            print("start of comparing to encodings")
            before_compare_encoding = time.time()
            matches = face_recognition.compare_faces(
                total_face_encodings, face_encoding, tolerance=0.55)
            after_compare_encoding = time.time()
            difference_of_time_encoding = after_compare_encoding - before_compare_encoding
            print("time to calculate comparison of encoding"+str(difference_of_time_encoding)) 

            # print(matches)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            before_face_distance = time.time() 
            face_distances = face_recognition.face_distance(
                total_face_encodings, face_encoding)
            
            
            best_match_index = np.argmin(face_distances)

            after_face_distance = time.time()

            time_distance = after_face_distance - before_face_distance

            print("time to calculate distance"+str(time_distance)) 
            time_now = time.time()
            #latency = float(time_of_frame) - float(time_now)
            #print("latency is"+str(latency))
            if matches[best_match_index]:
                name = total_face_names[best_match_index]
                #people_dict[str(name)] += 1  #increasing detect by 1
                
                if(int(people_dict[str(name)]) < 1):
                    #socket_test('UnLock')
                    print(str(name) + "has less than 2 detects,will unlock at 3")
                    print(people_dict[str(name)])
                    people_dict[str(name)] += 1  #increasing number of detects by 1
                    print("inside if")
                    continue
                else:
                    people_dict[str(name)] += 1 #increase detect by 1 , but take care
                    socket_test('UnLock')
                    print("UnLock")
                    unlock_time = time.time()
                    time_since_pi = time_of_frame - unlock_time
                    print("total time since pi sent frame to unlock"+ str(time_since_pi))
                    print("DETECTED NAME"+str(name)+"    SO UNLOCKING")
                    # reset the counter to 0.
                    people_dict[str(name)] = 0
                    continue
                    # return None
        before_lock_time = time.time()
        diff_time = before_lock_time - unlock_time
        print("this is diff time"+str(diff_time))
        if(diff_time > 5.00000):
        	socket_test('Lock')
        	print('Lock')
        	continue



def socket_test(send="Lock"):

    print("reached here")
    if ws:
        if(send == "Lock"):
            print("will lock now")
            ws.send(send)
        else:
            print("will unlock now")
            ws.send(send)


def writeFrame(frame, name):

    # write text onto the image and display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        print(name)
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 0.75, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imwrite('test1/Video' + str(count) + '.png', frame)


if __name__ == "__main__":
    # pick up the face encodings saved in the pickle file, which is saved as a dictionary.
    

    print("loading saved encodings")
    with open('saved_encoding/dataset.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)

    #video_capture = cv2.VideoCapture('http://118.185.61.234:8000/stream.mjpg')

    #video_capture = cv2.VideoCapture('rtmp://35.212.176.30:1935/myapp/example-stream')
    image_hub = imagezmq.ImageHub()
    queue = Queue()
    thread1 = Thread(target=threadFrameGet, args=("Thread-1", queue))
    thread2 = Thread(target=recognise_person, args=("Thread-2", queue))
    # thread3 = Thread(target=checkLock,args=("Thread-3"),queue)
    thread1.start()
    thread2.start()
    global WS
    global ws
    ws = create_connection(WS)
    socket_test('Lock')
    # thread3.start()
    # thread1.join()
    # thread2.join()
    # Get a reference to picam  ## also try using rtsp
    # process 1 should collect video frames and calculate its encoding.
    # process two must do all the matching using the frames collected
