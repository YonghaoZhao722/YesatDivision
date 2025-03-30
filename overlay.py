import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import cv2

# 文件路径
dic_path = '/Users/zhaoyonghao/Documents/MATLAB/DIC_mask/WT+pWL74 CLB2_Q570 MS2v6_Q670_1_DIC_s1.tif'
dapi_path = '/Users/zhaoyonghao/Documents/MATLAB/DAPI_processed/MAX_WT+pWL74 CLB2_Q570 MS2v6_Q670_1_w3DAPI-12-_s1.tif'
# dapi_path = '/Users/zhaoyonghao/Documents/MATLAB/CY3_processed/MAX_WT+pWL74 CLB2_Q570 MS2v6_Q670_1_w2CY3-100-_s1.tif'

# 读取图像
dic_image = io.imread(dic_path, as_gray=True)
dapi_image = io.imread(dapi_path, as_gray=True)

# 确保图像数据类型和范围正确 - 修复类型问题
dic_image_8bit = (dic_image * 255).astype(np.uint8)
dapi_image_8bit = (dapi_image * 255).astype(np.uint8)

try:
    # 预处理：增强图像对比度以提高特征点检测效果
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dic_enhanced = clahe.apply(dic_image_8bit)
    dapi_enhanced = clahe.apply(dapi_image_8bit)
    
    # 设置Lucas-Kanade参数
    lk_params = dict(
        winSize=(15, 15),       # 搜索窗口大小
        maxLevel=3,             # 金字塔层级
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # 终止条件
    )
    
    # 在DIC图像上检测特征点
    feature_params = dict(
        maxCorners=200,        # 最大角点数
        qualityLevel=0.01,     # 质量水平（越小检测的点越多）
        minDistance=7,         # 角点之间的最小距离
        blockSize=7            # 计算导数的邻域大小
    )
    
    # 对两个图像都检测特征点，可能会提高匹配效果
    prev_pts = cv2.goodFeaturesToTrack(dic_enhanced, **feature_params)
    
    if prev_pts is None or len(prev_pts) < 10:
        # 如果检测到的特征点太少，调整参数重试
        feature_params['qualityLevel'] = 0.005
        prev_pts = cv2.goodFeaturesToTrack(dic_enhanced, **feature_params)
    
    print(f"在DIC图像中检测到 {len(prev_pts)} 个特征点")
    
    # 使用光流法计算特征点在DAPI图像中的位置
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
        dic_enhanced, dapi_enhanced, prev_pts, None, **lk_params
    )
    
    # 仅保留成功跟踪的点
    good_old = prev_pts[status == 1]
    good_new = next_pts[status == 1]
    
    print(f"成功跟踪了 {len(good_new)} 个特征点")
    
    # 计算每个特征点的位移
    point_shifts = good_new - good_old
    
    # 使用RANSAC或中值滤波去除异常值，得到更稳健的估计
    # 这里使用中值作为平移估计
    median_shift = np.median(point_shifts, axis=0)
    tx, ty = -median_shift[0], -median_shift[1]
    
    print(f"估计的平移量: [x={tx}, y={ty}]")
    
    # 创建变换矩阵
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = tx
    warp_matrix[1, 2] = ty
    
    # 应用变换到DIC图像 - 修复：确保使用8bit图像
    height, width = dapi_image.shape
    aligned_dic = cv2.warpAffine(dic_image_8bit, warp_matrix, (width, height), flags=cv2.INTER_LINEAR)
    
    # 将结果转换回浮点数范围
    aligned_dic_float = aligned_dic.astype(float) / 255.0
    
    # 可视化跟踪点（用于调试）
    debug_img = cv2.cvtColor(dapi_image_8bit, cv2.COLOR_GRAY2BGR)
    for i, (old, new) in enumerate(zip(good_old, good_new)):
        a, b = old.ravel().astype(int)
        c, d = new.ravel().astype(int)
        
        # 绘制跟踪线和点
        cv2.line(debug_img, (a, b), (c, d), (0, 255, 0), 1)
        cv2.circle(debug_img, (a, b), 3, (0, 0, 255), -1)
        cv2.circle(debug_img, (c, d), 3, (255, 0, 0), -1)
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(dic_image, cmap='gray')
    axes[0, 0].set_title('DIC Image (Original)')
    
    axes[0, 1].imshow(dapi_image, cmap='gray')
    axes[0, 1].set_title('DAPI Image')
    
    # 特征点跟踪可视化
    axes[1, 0].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Feature Point Tracking')
    
    # 对齐结果
    axes[1, 1].imshow(dapi_image, cmap='gray')
    axes[1, 1].imshow(aligned_dic_float, cmap='jet', alpha=0.5)
    axes[1, 1].set_title('DAPI with Aligned DIC (Lucas-Kanade)')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    io.imsave('/Users/zhaoyonghao/Documents/MATLAB/aligned_dic_lk.tif', aligned_dic)
    
    # 可选：使用手动偏移值进行对比
    manual_shift = [-24, 17]  # 原始脚本中的手动偏移值 [y, x]
    
    # 创建手动偏移的变换矩阵
    manual_matrix = np.eye(2, 3, dtype=np.float32)
    manual_matrix[0, 2] = manual_shift[1]  # x轴偏移
    manual_matrix[1, 2] = manual_shift[0]  # y轴偏移
    
    # 应用手动变换 - 同样使用8bit图像
    manual_aligned_dic = cv2.warpAffine(dic_image_8bit, manual_matrix, (width, height), flags=cv2.INTER_LINEAR)
    manual_aligned_dic_float = manual_aligned_dic.astype(float) / 255.0
    
    # 比较自动和手动对齐的结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(dapi_image, cmap='gray')
    axes[0].imshow(aligned_dic_float, cmap='jet', alpha=0.5)
    axes[0].set_title('Lucas-Kanade Alignment')
    
    axes[1].imshow(dapi_image, cmap='gray')
    axes[1].imshow(manual_aligned_dic_float, cmap='jet', alpha=0.5)
    axes[1].set_title('Manual Alignment')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Lucas-Kanade对齐失败: {e}")
    
    # 尝试使用相位相关方法作为备选
    try:
        # 应用窗口函数减少边缘效应
        window = np.outer(np.hanning(dic_image.shape[0]), np.hanning(dic_image.shape[1]))
        dic_windowed = dic_image * window
        dapi_windowed = dapi_image * window
        
        # 使用相位相关方法计算平移
        shifts, error, diffphase = cv2.phaseCorrelate(
            dic_windowed.astype(np.float32), 
            dapi_windowed.astype(np.float32)
        )
        
        tx, ty = shifts
        
        print(f"相位相关估计的平移量: [x={tx}, y={ty}]")
        
        # 创建变换矩阵
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix[0, 2] = tx
        warp_matrix[1, 2] = ty
        
        # 应用变换到DIC图像 - 确保使用正确的数据类型
        height, width = dapi_image.shape
        aligned_dic = cv2.warpAffine(dic_image_8bit, warp_matrix, (width, height), flags=cv2.INTER_LINEAR)
        aligned_dic_float = aligned_dic.astype(float) / 255.0
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(dic_image, cmap='gray')
        axes[0].set_title('DIC Image (Original)')
        axes[1].imshow(dapi_image, cmap='gray')
        axes[1].set_title('DAPI Image')
        axes[2].imshow(dapi_image, cmap='gray')
        axes[2].imshow(aligned_dic_float, cmap='jet', alpha=0.5)
        axes[2].set_title('DAPI with Aligned DIC (Phase Correlation)')
        plt.show()
        
        # 保存结果
        io.imsave('/Users/zhaoyonghao/Documents/MATLAB/aligned_dic_phase.tif', aligned_dic)
    
    except Exception as e2:
        print(f"备选相位相关方法也失败: {e2}")
        print("建议使用手动对齐，可能原始图像特征不足以进行自动配准")