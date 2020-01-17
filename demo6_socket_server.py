import face_recognition
import cv2
from multiprocessing import Queue,Process
import asyncio
import websockets
import base64
import time

#抽取帧率检测 + 仅标记面部位置，+ socket服务器版本 进一步 减少检测压力

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
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
    start_server = websockets.serve(server_echo, "192.168.1.100", 8860)   #定义服务器信息
    asyncio.get_event_loop().run_until_complete(start_server)
    print('Websocket server start.')
    asyncio.get_event_loop().run_forever()


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
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), in zip(face_locations):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        print((left, top, right, bottom))

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    send_frame = image_to_base64(frame)  #把帧数据发出
    queue.put(send_frame)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

p.terminate()
p.join()
print('websocket server process is end')

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()