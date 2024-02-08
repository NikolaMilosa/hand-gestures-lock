import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import queue

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def main():
    cap = cv2.VideoCapture(0)
    qu = queue.Queue()
    last_gesture = "None"
    to_hit = ['Closed_Fist', 'Open_Palm', 'Closed_Fist', 'Open_Palm']
    attempt = []
    # Create a gesture recognizer instance with the live stream mode:
    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        qu.put((output_image.numpy_view(), result.gestures))

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    recognizer = GestureRecognizer.create_from_options(options)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit(1)
    
    current_timestamp = time.time()
    int_timestamp = int(current_timestamp)

    while True:
        ret, frame = cap.read()
    
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, int_timestamp)
        int_timestamp += 1

        frame, gestures = qu.get()
        cv2.imshow('Frame', frame)

        # if len(gestures) > 0 and gestures[0].category_name != last_gesture:
        #     last_gesture = gestures[0].category_name
        #     print("New state", last_gesture)

        if len(gestures) > 0 and gestures[0][0].category_name != last_gesture and gestures[0][0].category_name != "None":
            last_gesture = gestures[0][0].category_name
            print(last_gesture)
            attempt.append(last_gesture)
            if len(attempt) == len(to_hit):
                if attempt == to_hit:
                    print('Combination hit!')
                    break

                print('Combination missed...')
                attempt = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()