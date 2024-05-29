import cv2
import csv
import pandas as pd
import os

# CUDA를 사용하기 위한 import
cv2.setUseOptimized(True)
cv2.dnn_registerLayer('Crop', cv2.dnn.blobFromImage)
cv2.ocl.setUseOpenCL(True)

#새로운 파일 생성
ankle_csv_file = "extracted_keypoints/test.csv" 

#관절 좌표 데이터가 들어갈 데이터 형식 구성하기
ankle_csv = open(ankle_csv_file, 'w', newline='')
ankle_csv_writer = csv.writer(ankle_csv)

ankle_csv_writer.writerow(["Frame", "Head", "Neck", "RShoulder_X", "RShoulder_Y","LShoulder_X", "LShoulder_Y","RElbow_X", "RElbow_Y",
 "LElbow_X", "LElbow_Y", "RWrist_X", "RWrist_Y","LWrist_X", "LWrist_Y", "RHip_X", "RHip_Y","LHip_X", "LHip_Y","RKnee_X", "RKnee_Y",
 "LKnee_X", "LKnee_Y","RAnkle_X", "RAnkle_Y", "LAnkle_X", "LAnkle_Y","Chest"])

def output_keypoints(frame, net, threshold, BODY_PARTS, now_frame, total_frame):
    global points

    # 입력 이미지의 사이즈 정의
    image_height = 500
    image_width = 500

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    out_height = out.shape[2]
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    print(f"============================== frame: {now_frame:.0f} / {total_frame:.0f} ==============================")
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")
        
        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    return frame

def output_keypoints_with_lines(frame, POSE_PAIRS):
    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)

    return frame

def output_keypoints_with_lines_video(proto_file, weights_file, video_path, threshold, BODY_PARTS, POSE_PAIRS):

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    
    # GPU 사용
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv2.VideoCapture(video_path)

    while True:
        now_frame_boy = capture.get(cv2.CAP_PROP_POS_FRAMES)
        total_frame_boy = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        if now_frame_boy == total_frame_boy:
            break

        ret, frame_boy = capture.read()
        frame_boy = output_keypoints(frame=frame_boy, net=net, threshold=threshold, BODY_PARTS=BODY_PARTS, now_frame=now_frame_boy, total_frame=total_frame_boy)
        frame_boy = output_keypoints_with_lines(frame=frame_boy, POSE_PAIRS=POSE_PAIRS)
        frame_boy = cv2.resize(frame_boy, (500, 1000))
        cv2.imshow("Output_Keypoints", frame_boy)

        head = points[0] if points[0] else (None,None)
        neck = points[1] if points[1] else (None,None)
        r_shoulder = points[2] if points[2] else (None,None)
        l_shoulder = points[5] if points[5] else (None,None)
        r_elbow = points[3] if points[3] else (None,None)
        l_elbow = points[6] if points[6] else (None,None)
        r_wrist = points[4] if points[4] else (None,None)
        l_wrist = points[7] if points[7] else (None,None)
        r_hip = points[8] if points[8] else (None,None)
        l_hip = points[11] if points[11] else (None,None)
        r_knee = points[9] if points[9] else (None,None)
        l_knee = points[12] if points[12] else (None,None)
        r_ankle = points[10] if points[10] else (None,None)
        l_ankle = points[13] if points[13] else (None,None)
        chest = points[14] if points[14] else (None,None)

        #프레임마다 각 관절 별 x,y값 추가
        ankle_csv_writer.writerow([now_frame_boy, head[1], neck[1],
        r_shoulder[0], r_shoulder[1], l_shoulder[0], l_shoulder[1], r_elbow[0], r_elbow[1], l_elbow[0], l_elbow[1],
        r_wrist[0], r_wrist[1], l_wrist[0], l_wrist[1], r_hip[0], r_hip[1], l_hip[0], l_hip[1], r_knee[0], r_knee[1], l_knee[0], l_knee[1],
        r_ankle[0], r_ankle[1], l_ankle[0], l_ankle[1],chest[1]])
                            
        if cv2.waitKey(10) == 27:  
            break
    ankle_csv.close()
    capture.release()
    cv2.destroyAllWindows()

BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                  5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                  10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                  15: "Background"}

POSE_PAIRS_MPI = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                  [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

# 신경 네트워크의 구조를 지정하는 prototxt 파일 
protoFile_mpi = "prototxt_caffemodel/pose_deploy_linevec.prototxt"
protoFile_mpi_faster ="prototxt_caffemodel/pose_deploy_linevec_faster_4_stages.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_mpi = "prototxt_caffemodel/pose_iter_160000.caffemodel"

# 비디오 경로
video = "videos/single_unders/jw_type1_3.mp4"  

# 키포인트를 저장할 빈 리스트
points = []

output_keypoints_with_lines_video(proto_file=protoFile_mpi_faster, weights_file=weightsFile_mpi, video_path=video,
                                  threshold=0.1, BODY_PARTS=BODY_PARTS_MPI, POSE_PAIRS=POSE_PAIRS_MPI)
