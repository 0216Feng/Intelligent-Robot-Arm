import cv2
import numpy as np
from scipy.optimize import least_squares

class TiltCalibrator:
    def __init__(self, camera_matrix, dist_coeffs, H, h=0.0):
        """
        :param camera_matrix: 摄像头内参矩阵 3x3
        :param dist_coeffs: 畸变系数 1x5
        :param H: 摄像头安装高度（米）
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.H = H
        self.h = h  # 标定板平面与底座的高度差
        self.f = camera_matrix[0,0]  # 焦距（像素）

    def detect_board_center(self, img_path, pattern_size=(11,8)):
        """检测棋盘格中心坐标（图像坐标系原点在中心）"""
        # 检查文件是否存在
        import os
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
            
        # 读取并检查图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        # 输出图像信息
        print(f"处理图像: {img_path}, 尺寸: {img.shape}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 提高对比度（可选）
        # gray = cv2.equalizeHist(gray)
        
        # 创建调试目录
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        
        # 保存灰度图用于调试
        cv2.imwrite(f"{debug_dir}/gray_{os.path.basename(img_path)}", gray)
        
        # 尝试使用多种标志组合检测棋盘格角点
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        
        # 如果检测失败，尝试其他尺寸
        if not ret:
            print(f"在标准尺寸下检测失败: {pattern_size}")
            # 尝试其他可能的棋盘格尺寸
            alternative_sizes = [(12, 9), (10, 7), (9, 6), (8, 5), (7, 6), (6, 9)]
            
            for alt_size in alternative_sizes:
                print(f"尝试替代尺寸: {alt_size}")
                ret, corners = cv2.findChessboardCorners(gray, alt_size, flags)
                if ret:
                    pattern_size = alt_size
                    print(f"成功检测到棋盘格，使用尺寸: {pattern_size}")
                    break
        
        if not ret:
            # 保存调试图像，并添加更具体的错误信息
            failed_img = img.copy()
            cv2.putText(failed_img, "Detection Failed", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(f"{debug_dir}/failed_{os.path.basename(img_path)}", failed_img)
            raise ValueError(f"棋盘格检测失败: {img_path}")
        
        # 绘制检测到的角点并保存（用于调试）
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, ret)
        cv2.imwrite(f"{debug_dir}/corners_{os.path.basename(img_path)}", img_with_corners)
        
        # 亚像素精细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        # 计算中心点（图像坐标系原点在左上角）
        center_pixel = np.mean(corners, axis=0)[0]
        
        # 转换为以图像中心为原点 (u, v)
        h, w = img.shape[:2]
        u = center_pixel[0] - w/2
        v = h/2 - center_pixel[1]  # 注意Y轴方向
        
        # 在图像上标记中心点并保存
        cv2.circle(img_with_corners, (int(center_pixel[0]), int(center_pixel[1])), 
                5, (0, 0, 255), -1)
        cv2.imwrite(f"{debug_dir}/center_{os.path.basename(img_path)}", img_with_corners)
        
        print(f"成功检测棋盘格中心点: u={u:.1f}, v={v:.1f}")
        return u, v

    def theta_residuals(self, theta, D_ref_list, v_pixel_list):
        """最小二乘残差计算"""
        residuals = []
        for D_ref, v in zip(D_ref_list, v_pixel_list):
            alpha = np.arctan(v / self.f)
            H_eff = self.H + self.h  # 计算有效垂直距离
            pred_D = H_eff / np.tan(theta[0] + alpha)
            residuals.append(pred_D - D_ref)
        return np.array(residuals)

    def calibrate(self, img_paths, D_ref_list):
        """主标定函数"""
        # 1. 检测所有图像的标定板中心坐标
        v_pixel_list = []
        for path in img_paths:
            _, v = self.detect_board_center(path)
            v_pixel_list.append(v)
        
        # 2. 非线性最小二乘优化
        res = least_squares(
            self.theta_residuals, 
            x0=np.radians(30),  # 初始猜测角度（弧度）
            args=(D_ref_list, v_pixel_list),
            bounds=(0, np.pi/2) # 角度在0-90度之间
        )
        
        self.theta = res.x[0]
        return np.degrees(self.theta)

# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    # 摄像头参数示例（需替换为实际标定结果）
    camera_matrix = np.array([
        [2465.128986762485, 0, 984.7157440488494], 
        [0, 2465.442899090040, 540.7101182076190], 
        [0, 0, 1]
    ])
    dist_coeffs = [0.004625970045023, -0.0008902630420674212, -0.039376100278451, 0.563886970534354, 0]
    H = 17.0
    h = 1.0  

    # 初始化标定器
    calibrator = TiltCalibrator(camera_matrix, dist_coeffs, H)

    # 输入数据：图像路径列表 和 对应的真实距离列表
    img_paths = [
        "C:/Users/Feng/Pictures/Camera Roll/28.8.jpg", 
        "C:/Users/Feng/Pictures/Camera Roll/29.6.jpg",
        "C:/Users/Feng/Pictures/Camera Roll/30.0.jpg",
        "C:/Users/Feng/Pictures/Camera Roll/31.0.jpg",
        "C:/Users/Feng/Pictures/Camera Roll/31.8.jpg",
        "C:/Users/Feng/Pictures/Camera Roll/32.9.jpg",
        "C:/Users/Feng/Pictures/Camera Roll/34.7.jpg",
        "C:/Users/Feng/Pictures/Camera Roll/36.0.jpg",
        "C:/Users/Feng/Pictures/Camera Roll/37.2.jpg",
        "C:/Users/Feng/Pictures/Camera Roll/38.0.jpg"
    ]
    D_ref_list = [28.8, 29.6, 30.0, 31.0, 31.8, 32.9, 34.7, 36.0, 37.2, 38.0]  #cm

    # 执行标定
    theta_deg = calibrator.calibrate(img_paths, D_ref_list)
    print(f"标定结果：倾斜角θ = {theta_deg:.2f}°")