""" 人物去背及換背景 """


import cv2
import numpy as np
import streamlit as st
import os
# from PIL import Image
from datetime import datetime
# * 載入Mediapipe函式庫
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


st.title("圖像人物去背及換背景")
save_folder = "Remove or Replace background_saved"
os.makedirs(save_folder, exist_ok=True) # os.makedirs() 函數用於創建多層目錄
uploaded_file = st.file_uploader("1. 上傳圖片", type=["jpg", "png", "jpeg"])

# 將所有圖像處理邏輯，放在 if 裡面
if uploaded_file is not None:
    # 改用 getvalue() 避免 Streamlit 重新渲染時讀取到空 Buffer
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    imgBGR = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # 解碼成 NumPy 陣列 (BGR)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) # 轉換成 RGB 格式 
    original_img = imgRGB.copy() # 複製給 original_img，位於記憶體中的 RGB 格式 NumPy 陣列
    
    
    #原始上傳圖像
    st.subheader("原始上傳圖像")
    st.image(original_img, width="full") # st.image() 預設接受 RGB

      
    # * 人物去背
    model_path = "pose_landmarker_full.task" # 模型路徑
    BaseOptions = python.BaseOptions # 基本設定
    PoseLandmarker = vision.PoseLandmarker # 人體姿勢偵測與關鍵點定位的類別
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions # 偵測姿勢選項
    VisionRunningMode = vision.RunningMode # 執行模式
    # PoseLandmarkerOptions()可以設定偵測姿勢的基本設定, 包含偵測姿勢的模式、輸出的遮罩、姿勢名稱等
    options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, # VisionRunningMode.IMAGE代表偵測圖片
    output_segmentation_masks=True  #開啟人體遮罩
    )
    Media_rgb = original_img.copy() # 直接使用已解碼好的陣列
    if Media_rgb is None:
        raise ValueError("找不到圖片")
    h, w, _ = Media_rgb.shape # 取得圖片高、寬、通道，_表示不同通道
    bgr = cv2.cvtColor(Media_rgb, cv2.COLOR_RGB2BGR) 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Media_rgb) 
    with PoseLandmarker.create_from_options(options) as landmarker: 
        result = landmarker.detect(mp_image)
        if not result.segmentation_masks:
            st.write("未偵測到人體")
            st.stop()
        mask = result.segmentation_masks[0].numpy_view()
    

    # 最後面的 0 表示標準差（SigmaX = 0），自動計算高斯模糊的權重分配
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    condition = mask > 0.5 # 可以調整數字
    #以下為遮罩設定語法
    background = np.zeros_like(bgr) # 
    #以上為遮罩設定語法
    # ...（Ellipsis）代表「保留前面所有的維度」（即原本的高與寬）。
    # None（等同於 np.newaxis）代表「在最後面增加一個長度為 1 的新維度」。
    # 如果原本的 condition 形狀是 (1080, 1920)，經過 [..., None] 處理後，形狀會變成 (1080, 1920, 1)。這樣一來，它就能完美地自動複製並套用到彩色影像的 3 個通道（R、G、B）上。
    # NumPy 的條件選擇函式，當條件為 True 時，選擇 img，為 False 時，選擇 background
    # *依遮罩做影像合成； 將主體圖像與背景進行結合，可以得到影像合成的結果
    output = np.where(condition[..., None], bgr, background)


    # *將畫好框的 bgr 轉回 RGB，供 Streamlit 正確渲染顏色
    result_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    # 人物去背
    st.subheader("人物去背")
    st.image(result_rgb, width="full") # st.image() 預設接受 RGB
    # 人物去背自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Remove_background_filename = os.path.join(save_folder, f"Remove_background_{timestamp}.png")
    cv2.imwrite(Remove_background_filename, output)
    st.success(f"人物去背已經儲存 {Remove_background_filename}")

# * 換背景圖
    st.divider()
    background_file = st.file_uploader("2. 上傳待換背景圖片", type=["jpg", "png", "jpeg"])   
    # 將所有圖像處理邏輯，放在 if 裡面
    if background_file is not None:
    # 改用 getvalue() 避免 Streamlit 重新渲染時讀取到空 Buffer
    background_file_bytes = np.asarray(bytearray(background_file.getvalue()), dtype=np.uint8)
    imgBGR0 = cv2.imdecode(background_file_bytes, cv2.IMREAD_COLOR) # 解碼成 NumPy 陣列 (BGR)
    imgRGB0 = cv2.cvtColor(imgBGR0, cv2.COLOR_BGR2RGB) # 轉換成 RGB 格式 
    backgroundb_img = imgBGR0.copy() 
    backgroundr_img = imgRGB0.copy() 
    
    # 背景圖
    st.subheader("待換背景圖像")
    st.image(backgroundr_img, width="full") # st.image() 預設接受 RGB

    bg = cv2.resize(backgroundb_img, (w, h)) 
    background0 = bg
    
    # *依遮罩做影像合成； 將主體圖像與背景進行結合，可以得到影像合成的結果
    output0 = np.where(condition[..., None], bgr, background0)


    # *將畫好框的 bgr 轉回 RGB，供 Streamlit 正確渲染顏色
    result0_rgb = cv2.cvtColor(output0, cv2.COLOR_BGR2RGB)
    # 人物去背
    st.subheader("人物去背")
    st.image(result0_rgb, width=True) # st.image() 預設接受 RGB
    # 人物去背自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Replace_background_filename = os.path.join(save_folder, f"Replace_background_{timestamp}.png")
    cv2.imwrite(Replace_background_filename, output0)
    st.success(f"更換背景圖已經儲存 {Replace_background_filename}")