import face_recognition
import cv2
import numpy as np

#Bilgisayarın Kamerasını açar.
video_capture = cv2.VideoCapture(0)

# Örnek bir resim yükleyin ve onu nasıl tanıyacağınızı öğrenin.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

jordan_image = face_recognition.load_image_file("jordan.png")
jordan_face_encoding = face_recognition.face_encodings(jordan_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Bilinen yüz kodlama dizilerini ve adlarını oluşturun
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    jordan_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Micheal Jordan"
]

while True:
    # Tek bir video karesi yakalayın
    ret, frame = video_capture.read()

    # Görüntüyü BGR renginden (OpenCV'nin kullandığı) RGB rengine (face_recognition'ın kullandığı) dönüştürün
    rgb_frame = frame[:, :, ::-1]

    # Video çerçevesindeki tüm yüzleri ve yüz kodlamalarını bulun
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Bu video çerçevesindeki her bir yüz boyunca döngü yapın
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Yüzün bilinen yüzlerle eşleşip eşleşmediğine bakın
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Yüzün etrafına bir kutu çizin
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Yüzün altına bir ad içeren bir etiket çizin
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Ortaya çıkan görüntüyü göster
    cv2.imshow('Video', frame)

    # Çıkmak için q tuşuna basabilirsin
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()