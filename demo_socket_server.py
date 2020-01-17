import face_recognition
import cv2
import numpy as np
import websockets
from multiprocessing import Queue,Process
import asyncio
import websockets
import base64
import time

#websocket_server版本

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
video_capture.set(3,800) #设置分辨率
video_capture.set(4,600)

imgdir = "./"
# # Load a second sample picture and learn how to recognize it. sjj
# biden_image = face_recognition.load_image_file("sjj.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# # Load a second sample picture and learn how to recognize it. 刘嘉印
# liujiayin_image = face_recognition.load_image_file("liujiayin.jpg")
# liujiayin_face_encoding = face_recognition.face_encodings(liujiayin_image)[0]

# # Load a second sample picture and learn how to recognize it. 朗晟
# langsheng_image = face_recognition.load_image_file("/home/fish/project/face/img/langsheng/5.jpg")
# langsheng_face_encoding = face_recognition.face_encodings(langsheng_image)[0]

# Create arrays of known face encodings and their names
# known_face_encodings = [
#     biden_face_encoding,
#     liujiayin_face_encoding,
#     langsheng_face_encoding
# ]

# known_face_names = [
#     "Sun Jun Jian",
#     "Liu Jia Yin",
#     "Lang Sheng"
# ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
frame_pre_count = 1      #检测帧
frame_size = 0.5         #缩放比率，一定要可以被1整除 
frame_count = 0

queue = Queue(1) #进程间通信队列  最大1
#服务器
async def server_echo(websocket, path):
    while True:
        if not queue.empty():
            time.sleep(0.1) #休眠一点时间，让帧来得及处理
            send_frame = queue.get()
            #print(send_frame)
            await websocket.send(send_frame)

def server_run(name, queue):
    print('Websocket server start.')
    asyncio.get_event_loop().run_forever()

start_server = websockets.serve(server_echo, "192.168.1.103", 8860)   #定义服务器信息
asyncio.get_event_loop().run_until_complete(start_server)

p = Process(target=server_run, args=('video_websocket_server', queue))
print('Child process will start.')
p.start()

#图片转base64
def image_to_base64(image_np):
    image = cv2.imencode('.jpg',image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    #print(image_code)
    return image_code



while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if frame_count < frame_pre_count:    #3帧检测一次
        process_this_frame = False
        frame_count += 1
    else:
        process_this_frame = True
        frame_count = frame_count % frame_pre_count
        #print(process_this_frame)


    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=frame_size, fy=frame_size)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        #face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_encodings = []

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.4)
            matches = [False]
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= int(1/frame_size)
        right *= int(1/frame_size)
        bottom *= int(1/frame_size)
        left *= int(1/frame_size)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    send_frame = image_to_base64(frame)  #把帧数据发出
    queue.put(send_frame)
    # Display the resulting image
    cv2.imshow('VIDEO', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

p.terminate()
p.join()
print('websocket server process is end')

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
