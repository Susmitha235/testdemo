import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, redirect

app = Flask(__name__, template_folder='templates')
camera = None  # Global variable to hold the camera object

# MediaPipe FaceMesh setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Function to provide camera image as a response
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global camera  # Use the global camera object

    # Open the camera if it is not already open
    if camera is None:
        camera = cv2.VideoCapture(0)  # Replace with the appropriate camera index or video file path

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while True:
            if camera is None:
                break

            ret, frame = camera.read()
            if not ret:
                break

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)

            # Draw the face mesh annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            # Flip the image horizontally for a selfie-view display.
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Release the camera when finished
    if camera is not None:
        camera.release()
        camera = None  # Set the camera object back to None


@app.route('/start_camera')
def start_camera():
    global camera  # Use the global camera object

    if camera is None:
        camera = cv2.VideoCapture(0)  # Replace with the appropriate camera index or video file path

    return "Camera started"


if __name__ == '__main__':
    app.run(port=5001)
