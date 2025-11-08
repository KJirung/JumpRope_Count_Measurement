# Jump Rope Exercise Assistance Program

<p align="center">
  <b>Jin-Woong Kim<sup>1</sup> · Jae-Woo Shin<sup>2</sup> · Seoung-Ho Choi<sup>3*</sup></b><br>
  <sup>1</sup>Department of Convergence IT Engineering, Hansung University, Seoul, Republic of Korea    
  <sup>2</sup>Department of IT Business Administration, Hanshin University, Osan-si, Gyeonggi-do, Republic of Korea    
  <sup>3</sup>College of Liberal Arts, Faculty of Basic Liberal Arts, Hansung University, Seoul, Republic of Korea
</p>

****


[![Paper](https://img.shields.io/badge/PAPER-IEEE_Access_Open_Access-E84C3D?style=for-the-badge)](https://doi.org/10.1109/ACCESS.2024.3496510)
[![Journal](https://img.shields.io/badge/JOURNAL-IEEE_Access-0A66C2?style=for-the-badge)](https://ieeeaccess.ieee.org/)
[![Publisher](https://img.shields.io/badge/PUBLISHER-IEEE-black?style=for-the-badge)](https://www.ieee.org/)

<img width="1200" alt="Architecture" src="https://github.com/user-attachments/assets/00937f9f-18ca-45b3-ad65-22038bddd07c" />
<p align="center"><em>Figure 1. Overall architecture of the proposed jump-rope exercise assistance program using video-based pose estimation and AI models.</em></p>

## Abstract

Jump rope exercise requires rapid tempo and rhythmic breathing, often causing users to lose count of their repetitions during workouts. To address this issue, we propose an intelligent jump-rope assistance program that recognizes motion patterns and analyzes the influence of joint coordinates on these movements. The system extracts frame-wise joint coordinate data from exercise videos using the MPII model of OpenPose, applies Min-Max scaling and missing-value interpolation, and classifies motions through five machine-learning (RF, Extra Trees, CatBoost, LightGBM, XGBoost) and two deep-learning models (LSTM, Transformer). Moving-average smoothing is introduced to reduce noise in repetition counting. To interpret motion dynamics, SHAP analysis identifies key joint coordinates, and Odds-Ratio analysis quantifies how joint positions affect airborne or grounded states. Experimental results confirm the proposed framework’s accuracy and interpretability for motion recognition and repetition counting.

## Motivation

- Counting repetitions manually during fast jump-rope motion is inaccurate and hinders exercise tracking.  
- Previous sensor-based methods (e.g., smartwatches) incur cost and environment limitations.  
- A vision-based AI approach enables real-time analysis without specialized equipment.  
- Understanding **which joints** drive correct jump motions supports exercise feedback and injury prevention.

## Contribution

- Proposed a **video-based AI system** that recognizes jump-rope motions and measures repetition counts automatically.  
- Collected a **100-video dataset** (10 performers, single & double types) and extracted frame-level joint coordinates via OpenPose (MPII).  
- Evaluated **seven models** (5 machine-learning + 2 deep-learning) with accuracy, precision, recall, F1, AUROC, and AP metrics.  
- Applied a **moving-average filter** to suppress prediction noise and stabilize jump-count estimation.  
- Employed **SHAP analysis** to identify key joint-coordinate influences and **Odds-Ratio analysis** to quantify occurrence probabilities.  
- Demonstrated that the Transformer (single type) and LightGBM (double type) achieved the best performance with interpretable results.

## Citation

If you use this work, please cite:

```bibtex
@article{kim2024jumprope,
  title={Jump Rope Exercise Assistance Program},
  author={Jin-Woong Kim and Jae-Woo Shin and Seoung-Ho Choi},
  journal={IEEE Access},
  volume={12},
  pages={169149--169162},
  year={2024},
  doi={10.1109/ACCESS.2024.3496510}
}
