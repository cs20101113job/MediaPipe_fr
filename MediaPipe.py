# 分別利用KNN/ SVM/ Mediapipe三種模型進行人臉偵測
# 整體流程包含：圖片上傳 --＞格式解碼 --＞ KNN 模型辨識人臉 --＞ 繪製標籤與外框 --＞ Streamlit 視覺化與自動存檔。
import cv2
import numpy as np
import streamlit as st
import os
from PIL import Image
from datetime import datetime


# 載入Mediapipe函式庫
import mediapipe as mp
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision

st.title("KNN/ SVM/ Mediapipe三種模型進行人臉偵測")
save_folder = "KNN_SVM_Mediapipe_face_recognition_saved"
os.makedirs(save_folder, exist_ok=True) # os.makedirs() 函數用於創建多層目錄
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
# 將所有圖像處理邏輯，放在 if 裡面
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imgBGR = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # 解碼成 NumPy 陣列 (BGR)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) # 轉換成 RGB 格式 
    original_img = imgRGB.copy() # 複製給 original_img，位於記憶體中的 RGB 格式 NumPy 陣列
    
    
    #原始上傳圖像
    st.subheader("原始上傳圖像")
    st.image(original_img, width="stretch") # st.image() 預設接受 RGB

    
    
   
    # ---MediaPipe人臉偵測-用RGB---
   
    base_options = python.BaseOptions(model_asset_path="face_detector.task")
    options = vision.FaceDetectorOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
    detector = vision.FaceDetector.create_from_options(options) 
    # *直接使用記憶體中的 original_img，不調用 cv2.imread
    Media_rgb = original_img.copy() # 直接使用已解碼好的陣列
    bgr = cv2.cvtColor(Media_rgb, cv2.COLOR_RGB2BGR) 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Media_rgb) 
    result = detector.detect(mp_image)
    for detection in result.detections: # result.detections 是一個列表，包含了所有檢測到的人臉資訊。這裡使用 for 迴圈來遍歷每個檢測到的人臉（detection），以便對每個人臉進行後續處理，例如繪製邊界框和顯示分數。
        bbox = detection.bounding_box # detection.bounding_box 是一個 BoundingBox 類別的實例，包含了該人臉的邊界框資訊。這裡將其賦值給 bbox 變數，以便後續使用。
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height # 從邊界框中提取出左上角的座標和寬度、高度。
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2) #第2點右下角(x + w, y + h)，(0, 255, 0) BGR是藍色，2是線寬
        score = detection.categories[0].score # detection.categories[0].score 是一個浮點數(例如：0.925)，表示該人臉檢測的信心分數（Confidence Score），範圍通常在 0 到 1 之間。這裡取[0]第一個分類（通常是人臉）的分數，並將其賦值給 score 變數。
        score_text = f"{int(score * 100)}%" # 將 0.925 乘以 100 得到 92.5，再用 int() 去除小數點變成 92。
        cv2.putText(bgr, score_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # 文字放在左上角座標 (x, y - 10) 的位置，用了 Hershey Simplex 字體，字體大小為 0.8，顏色為綠色 (0, 255, 0)，線寬為 2。
    # *將畫好框的 bgr 轉回 RGB，供 Streamlit 正確渲染顏色
    result_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # MediaPipe人臉偵測
    st.subheader("MediaPipe人臉偵測")
    st.image(result_rgb, width="stretch") # st.image() 預設接受 RGB
    # MediaPipe自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MediaPipe_filename = os.path.join(save_folder, f"MediaPipe_{timestamp}.png")
    cv2.imwrite(MediaPipe_filename, bgr)
    st.success(f"MediaPipe人臉偵測已經儲存 {MediaPipe_filename}")

    
 