# 資訊保護
from dotenv import load_dotenv
import os
# opencv
import cv2

# 載入.env檔案
load_dotenv('private_info.env') 

# 監視器IP
ip_address = os.getenv("CAM1_IP")

# RTSP 帳戶
rtsp_account = os.getenv("RTSP_ID")
rtsp_password = os.getenv("RTSP_PWD")

# 使用OpenCV連接監視器
cap = cv2.VideoCapture(f"rtsp://{rtsp_account}:{rtsp_password}@{ip_address}:554/stream2")
print(cap)
# 確認連接
if not cap.isOpened():
    print("無法連接監視器")
    exit()

# 顯示畫面
while True:
    ret, frame = cap.read()
    cv2.imshow('IP Camera', frame)

    # 按下 'q' 退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean
cap.release()
cv2.destroyAllWindows()