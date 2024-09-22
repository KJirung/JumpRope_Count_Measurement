from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import CCA
import scipy.stats as stats
import numpy as np
import pandas as pd
from scipy.stats import linregress

train_total = pd.read_csv('final_dataset/double/double_train.csv')

result_df = pd.DataFrame(columns=['Feature1', 'Feature2', 'Pearson_Corr', 'Pearson_p-value', 'Spearman_Corr', 'Spearman_p-value', 'Kendall_Corr', 'Kendall_p-value'])

# 27개 특성 목록
features = ['Head', 'Neck', 'RShoulder_X', 'RShoulder_Y', 'LShoulder_X',
           'LShoulder_Y', 'RElbow_X', 'RElbow_Y', 'LElbow_X', 'LElbow_Y',
           'RWrist_X', 'RWrist_Y', 'LWrist_X', 'LWrist_Y', 'RHip_X', 'RHip_Y',
           'LHip_X', 'LHip_Y', 'RKnee_X', 'RKnee_Y', 'LKnee_X', 'LKnee_Y',
           'RAnkle_X', 'RAnkle_Y', 'LAnkle_X', 'LAnkle_Y', 'Chest']

# 특성 조합 생성
feature_pairs = list(combinations(features, 2))

# CCA 및 상관관계 분석
result_list = []
for X1, X2 in feature_pairs:
    X1_data = train_total[[X1]]
    X2_data = train_total[[X2]]
    
    scaler_X1 = MinMaxScaler()
    scaler_X2 = MinMaxScaler()
    
    X1_sc = scaler_X1.fit_transform(X1_data)
    X2_sc = scaler_X2.fit_transform(X2_data)
    
    cca = CCA(scale=False, n_components=1)
    cca.fit(X1_sc, X2_sc)
    X1_c, X2_c = cca.transform(X1_sc, X2_sc)
    
    corr_coefficient, p_value = np.corrcoef(X1_c[:, 0], X2_c[:, 0])[0, 1], linregress(X1_c[:, 0], X2_c[:, 0]).pvalue
    
    rho, p_val = stats.pearsonr(X1_c[:, 0], X2_c[:, 0])
    rho_spearman, p_val_spearman = stats.spearmanr(X1_c[:, 0], X2_c[:, 0])
    rho_kendall, p_val_kendall = stats.kendalltau(X1_c[:, 0], X2_c[:, 0])
    
    result_list.append([X1, X2, rho, p_val, rho_spearman, p_val_spearman, rho_kendall, p_val_kendall])

result_df = pd.DataFrame(result_list, columns=['Feature1', 'Feature2', 'Pearson_Corr', 'Pearson_p-value', 'Spearman_Corr', 'Spearman_p-value', 'Kendall_Corr', 'Kendall_p-value'])

# CSV 파일 저장
result_df.to_csv('cca_results/cca_double.csv', index=False)
