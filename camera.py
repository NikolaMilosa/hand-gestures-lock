import cv2
import mediapipe as mp
import time
import queue
import argparse
import logging
import os

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def get_logger(file):
    filename = os.path.expanduser(file)
    file_dir, _ = os.path.split(filename)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    FORMAT = "[%(asctime)s] %(levelname)-8s %(message)s"
    filemode = 'a'
    datefmt = "%H:%M:%S"
    
    file_handler = logging.FileHandler(filename, filemode)
    file_handler.setFormatter(logging.Formatter(FORMAT, datefmt=datefmt))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(FORMAT, datefmt=datefmt))

    logger = logging.getLogger("unlocker")
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser("hand-gestures-unlocker", description="A small program that can be setup to run and login with hand gestures")
    parser.add_argument("--combination", dest="combination", type=str, help="Comma separated values of gestures to use as combination", required=True)
    parser.add_argument("--camera", dest="camera", type=int, help="A camera to use. Should be an integer, by default will use 0", default=0)
    parser.add_argument("--log-file", dest="log_file", type=str, help="Log file to write to. By default will write to ~/unlocker", default="~/unlocker/log")

    return parser.parse_args()

def check_combination(combination, logger):
    known = {
        'closed': 'Closed_Fist',
        'open': 'Open_Palm',
        'pointing': 'Pointing_Up',
        'thumbs_down': 'Thumbs_Down',
        'thumbs_up': 'Thumbs_Up',
        'victory': 'Victory',
        'love': 'ILoveYou'
    }

    combination = [elem.strip() for elem in combination.split(',')]
    if len(combination) <= 3:
        logger.error("Combantion too simple. Provide more than 3 simbols")
        exit(1)

    if not all([elem in known for elem in combination]):
        logger.error("Not all tokens are in combination")
        logger.error("Expected keys are: %s", ','.join(known.keys()))
        exit(1)

    return [known[elem] for elem in combination]

def check_camera(camera, logger):
    cap = cv2.VideoCapture(camera)
    if cap is None or not cap.isOpened():
        logger.error("Camera not accessible")
        exit(1)

    return cap

def main():
    args = parse_args()
    logger = get_logger(args.log_file)
    to_hit = check_combination(args.combination, logger)
    cap = check_camera(args.camera, logger)

    qu = queue.Queue()
    last_gesture = "None"
    attempt = []
    # Create a gesture recognizer instance with the live stream mode:
    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        qu.put((output_image.numpy_view(), result.gestures))

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    recognizer = GestureRecognizer.create_from_options(options)
    
    current_timestamp = time.time()
    int_timestamp = int(current_timestamp)

    while True:
        ret, frame = cap.read()
    
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, int_timestamp)
        int_timestamp += 1

        frame, gestures = qu.get()
        cv2.imshow('Frame', frame)

        if len(gestures) > 0 and gestures[0][0].category_name != last_gesture and gestures[0][0].category_name != "None":
            last_gesture = gestures[0][0].category_name
            logger.info(last_gesture)
            attempt.append(last_gesture)
            if len(attempt) == len(to_hit):
                if attempt == to_hit:
                    logger.info('Combination hit!')
                    break

                logger.warn('Combination missed... Sleeping 3s')
                time.sleep(3)
                attempt = []
                last_gesture = "None"

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Succeeded")

if __name__ == "__main__":
    main()