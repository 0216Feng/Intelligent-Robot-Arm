import cv2  
import numpy as np  
#IntrinsicMatrix
fx,cx,fy,cy=2465.128986762485,984.7157440488494,2465.442899090040,540.7101182076190
#TangentialDistortion
p1,p2=0.004625970045023,-0.0008902630420674212
#RadialDistortion
k1,k2,k3=-0.039376100278451,0.563886970534354,0

def undistort_video(camera_matrix, dist_coeffs, video_source=1):  
    # 打开视频流  
    cap = cv2.VideoCapture(video_source)  
  
    # 读取第一帧以获取其尺寸  
    ret, frame = cap.read()  
    if not ret:  
        print("无法打开视频流或文件")  
        return  
  
    h, w = frame.shape[:2]  
  
    # 创建一个窗口用于显示校正前的视频  
    cv2.namedWindow('Original Video', cv2.WINDOW_AUTOSIZE)  
  
    # 创建一个窗口用于显示校正后的视频  
    cv2.namedWindow('Undistorted Video', cv2.WINDOW_AUTOSIZE)  
  
    while True:  
        # 读取视频帧  
        ret, frame = cap.read()  
        if not ret:  
            break  
  
        # 校正图像  
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)  
  
        # 显示校正前和校正后的视频  
        cv2.imshow('Original Video', frame)  
        cv2.imshow('Undistorted Video', undistorted_frame)  
  
        # 按'q'键退出  
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
  
    # 释放资源和关闭窗口  
    cap.release()  
    cv2.destroyAllWindows()  
  
# 假设你已经有了这些参数，这里只是示例值  
camera_matrix = np.array([[fx, 0, cx],  
                          [0, fy, cy],  
                          [0, 0,  1]], dtype=np.float32)  
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)  
  
# 注意：将 fx, fy, cx, cy, k1, k2, p1, p2, k3 替换为你的实际标定值  
  
# 调用函数  
undistort_video(camera_matrix, dist_coeffs)