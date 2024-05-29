import os
import pandas as pd

# 폴더 경로 설정
folder_path = 'extracted_keypoints/quickly_train'

# 모든 CSV 파일을 저장할 빈 DataFrame 생성
combined_csv = pd.DataFrame()

# 폴더 내의 모든 파일을 확인하며 CSV 파일인 경우 합치기
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # CSV 파일을 DataFrame으로 읽기
        df = pd.read_csv(file_path)
        # 현재 파일의 데이터를 빈 DataFrame에 추가
        combined_csv = pd.concat([combined_csv, df], ignore_index=True)

# 결과를 하나의 CSV 파일로 저장
combined_csv.to_csv('merged_dataset/double_train/double_train.csv', index=False)

