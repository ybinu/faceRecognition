import face_recognition

#图片版本
image = face_recognition.load_image_file("timg.jpg")
face_locations = face_recognition.face_locations(image)

print(face_locations)