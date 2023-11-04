import os
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2
from matplotlib import pyplot as plt

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = mp.tasks.BaseOptions

# グータッチが行われているかのしきい値
threshold_distance_of_gou = 0.1

# デバッグモードにするか（入力の画像や処理後の画像を表示するか）
is_local_debug = False

# 手に関する情報が含まれている配列のインデックス
left_hand_index = [15,17,19,21]
right_hand_index = [16,18,20,22]

def plot_MOT16_image(img):
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img0), plt.xticks([]), plt.yticks([])
    plt.show()
    print('MOT16 - https://motchallenge.net/data/MOT16/')



def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# 入力された手の距離を計算する
def cal_distance(hand_1,hand_2):
    tmp = np.sqrt((hand_1[0] - hand_2[0])**2 + (hand_1[1] - hand_2[1])**2)
    if(is_local_debug):
        print(tmp)
    return tmp

# 画像の人物がグータッチしているか判定する
def check_gou_touch (detection_result):
    is_client_debug = os.environ.get('IS_CLIENT_DEBUG', False)
    pose_landmarks_list = detection_result.pose_landmarks
    if(is_client_debug or len(pose_landmarks_list) >= 2):
        first_person_pose_landmarks = pose_landmarks_list[0]
        x = 0
        y = 0
        for i in right_hand_index:
            x += first_person_pose_landmarks[i].x
            y += first_person_pose_landmarks[i].y
        first_person_right_position = [x/4,y/4]
        x = 0
        y = 0
        for i in left_hand_index:
            x += first_person_pose_landmarks[i].x
            y += first_person_pose_landmarks[i].y
        first_person_left_position = [x/4,y/4]
        second_person_pose_landmarks = pose_landmarks_list[1]
        x = 0
        y = 0
        for i in right_hand_index:
            x += second_person_pose_landmarks[i].x
            y += second_person_pose_landmarks[i].y
        second_person_right_position = [x/4,y/4]
        x = 0
        y = 0
        for i in left_hand_index:
            x += second_person_pose_landmarks[i].x
            y += second_person_pose_landmarks[i].y
        second_person_left_position = [x/4,y/4]

        distance = []
        distance.append(cal_distance(first_person_right_position, second_person_right_position))
        distance.append(cal_distance(first_person_right_position, second_person_left_position))
        distance.append(cal_distance(first_person_left_position, second_person_right_position))
        distance.append(cal_distance(first_person_left_position, second_person_left_position))
        distance.append(cal_distance(first_person_left_position,first_person_right_position))
        distance.append(cal_distance(second_person_left_position, second_person_right_position))
        if is_client_debug:
            return {"num_of_people": len(pose_landmarks_list),"distance":distance}
        for i in distance:
            if i <= threshold_distance_of_gou:
                return True
    return False



# 送信された画像をmediapipeにかける
def image_detector(img_path):
    img = cv2.imread(img_path)

    if(is_local_debug):
        plot_MOT16_image(img)

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=2,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(img_path)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    if (is_local_debug):
    # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        plot_MOT16_image(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    return detection_result

def change_client_debug_mode(bool):
        os.environ['IS_CLIENT_DEBUG'] = bool
        return {'message':'now mode ' + bool}

def check_client_debug_mode():
    return {'message':'now mode ' + str(os.environ.get('IS_CLIENT_DEBUG', False))}