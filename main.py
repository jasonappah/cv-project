import freenect
import cv2
import numpy as np
import face_recognition

video_capture = cv2.VideoCapture(1)

mason_image = face_recognition.load_image_file("faces/mason.png")
mason_face_encoding = face_recognition.face_encodings(mason_image)[0]

kynlee_image = face_recognition.load_image_file("faces/kynlee.png")
kynlee_face_encoding = face_recognition.face_encodings(kynlee_image)[0]

# carrie_image = face_recognition.load_image_file("faces/carrie.png")
# carrie_face_encoding = face_recognition.face_encodings(carrie_image)[0]

# mike_image = face_recognition.load_image_file("faces/mike.png")
# mike_face_encoding = face_recognition.face_encodings(mike_image)[0]

known_face_encodings = [
    mason_face_encoding,
    kynlee_face_encoding,
    # carrie_face_encoding,
    # mike_face_encoding
]
known_face_names = [
    "Mason Thomas - mgt210000",
    "Kynlee Thomas",
    # "Carrie Thomas",
    # "Michael Thomas"
]

def get_video():
    frame, _ = freenect.sync_get_video()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def get_depth_frame():
    frame, _ = freenect.sync_get_depth()
    return frame

def get_depth_at_point(frame, x: int, y:int):
    
    depth = frame[y, x]
    return depth


while True:
    depth_frame = get_depth_frame()
    rgb = get_video()
    ret, frame = video_capture.read()
    
    rgb_frame = frame[:, :, ::-1]
    small = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
    
    face_locations = face_recognition.face_locations(small)
    face_encodings = face_recognition.face_encodings(small, face_locations)
    
       # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('RGB', rgb)
    cv2.imshow('Depth', depth_frame / 2048)  # simple visualization
    cv2.imshow('Video', frame)
    depth = get_depth_at_point(depth_frame, 223, 313)
    
    # print(depth)
    
    if 890 > depth > 870:
        print("no drawer open")
    if 871 > depth > 850:
        print("bottom drawer open")
    if 851 > depth > 830:
        print("middle drawer open")
    if 831 > depth > 800:
        print("top drawer open")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

