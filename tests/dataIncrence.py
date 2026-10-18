import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class WaferWhiteSpotDetector:
    """
    晶圆异常白点检测器
    适配长鑫半导体晶圆检测场景，区分规则器件白点与异常白点
    """
    def __init__(self, config: Dict = None):
        """
        初始化检测器，配置参数均为晶圆场景工程优化值
        """
        self.config = {
            # 预处理参数
            "median_blur_ksize": 3,  # 中值滤波核大小
            "clahe_clip": 2.0,  # CLAHE对比度限制
            "clahe_tile": (8, 8),  # CLAHE分块大小
            # 分割参数
            "adaptive_block": 15,  # 自适应阈值块大小
            "adaptive_c": 3,  # 自适应阈值偏移量
            # 形态学参数
            "morph_kernel_size": 3,  # 形态学核大小
            # 筛选参数
            "min_area": 2,  # 最小面积阈值，像素，过滤噪声
            "max_area": 1000,  # 最大面积阈值，像素
            "circularity_threshold": 0.6,  # 圆形度阈值，大于为规则，小于为异常
            "area_std_factor": 2.0,  # 面积统计置信区间系数，±2σ
            # 标注参数
            "normal_color": (0, 255, 0),  # 正常白点：绿色
            "abnormal_color": (0, 0, 255),  # 异常白点：红色
            "contour_thickness": 2,  # 轮廓线宽
            # 峰值校验
            "peak_check_kernel": 3  # 局部峰值校验窗口
        }
        if config:
            self.config.update(config)

        # 形态学核
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config["morph_kernel_size"], self.config["morph_kernel_size"])
        )

    def _read_image(self, img_path: str) -> np.ndarray:
        """
        读取晶圆图像，支持fts/常见图像格式，保留原始位深
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像不存在: {img_path}")

        # 读取原始位深
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        # 统一转为灰度图，三通道转单通道
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        预处理：中值去噪 + 对比度增强
        """
        # 中值滤波去颗粒噪声
        img_med = cv2.medianBlur(img, self.config["median_blur_ksize"])

        # CLAHE自适应对比度增强，解决暗区漏检
        # 16位图转8位处理CLAHE，16位CLAHE效果不稳定
        if img.dtype == np.uint16:
            img_8 = cv2.normalize(img_med, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            img_8 = img_med.copy()

        clahe = cv2.createCLAHE(
            clipLimit=self.config["clahe_clip"],
            tileGridSize=self.config["clahe_tile"]
        )
        img_enhance = clahe.apply(img_8)
        return img_enhance

    def _local_peak_check(self, img: np.ndarray, centroids: np.ndarray) -> List[bool]:
        """
        局部峰值校验：验证白点是否为局部灰度极大值，过滤亮线/亮边误判
        实现思路：取白点周围局部像素窗口，比较中心与窗口峰值
        """
        k = self.config["peak_check_kernel"]
        half_k = k // 2
        h, w = img.shape
        peak_flags = []

        for (x, y) in centroids:
            x, y = int(round(x)), int(round(y))
            # 边界处理
            x1 = max(0, x - half_k)
            x2 = min(w, x + half_k + 1)
            y1 = max(0, y - half_k)
            y2 = min(h, y + half_k + 1)

            local_region = img[y1:y2, x1:x2]
            center_val = img[y, x]
            # 中心是局部最大值则为有效白点
            is_peak = center_val >= local_region.max()
            peak_flags.append(is_peak)

        return peak_flags

    def _classify_spots(self, stats: np.ndarray, centroids: np.ndarray, peak_flags: List[bool]) -> Tuple[
        List[Dict], List[bool]]:
        """
        区分正常器件白点与异常白点
        规则：1. 局部峰值校验通过 2. 面积在统计置信区间内 3. 形状规则度达标
        """
        # 提取面积、质心
        areas = stats[:, cv2.CC_STAT_AREA]
        spot_list = []
        is_abnormal_list = []

        # 面积统计：过滤极小极大后计算均值标准差
        valid_areas = areas[
            (areas > self.config["min_area"]) &
            (areas < self.config["max_area"])
            ]
        area_mean = np.mean(valid_areas) if len(valid_areas) > 0 else 0
        area_std = np.std(valid_areas) if len(valid_areas) > 0 else 1e-6

        for i in range(1, len(stats)):  # 0是背景标签
            area = areas[i]
            cx, cy = centroids[i]
            is_peak = peak_flags[i - 1]

            # 基础过滤：面积范围、峰值校验
            if area < self.config["min_area"] or area > self.config["max_area"] or not is_peak:
                is_abnormal = True
            else:
                # 面积偏离度：偏离±2σ判定为面积异常
                area_deviation = abs(area - area_mean) / area_std
                # 形状规则度：外接矩形长宽比，接近1为圆形
                w_rect = stats[i, cv2.CC_STAT_WIDTH]
                h_rect = stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = min(w_rect, h_rect) / (max(w_rect, h_rect) + 1e-6)

                # 异常判定：面积偏离大 或 形状不规则
                area_abnormal = area_deviation > self.config["area_std_factor"]
                shape_abnormal = aspect_ratio < self.config["circularity_threshold"]
                is_abnormal = area_abnormal or shape_abnormal

            spot_list.append({
                "spot_id": i,
                "center_x": round(cx, 2),
                "center_y": round(cy, 2),
                "area_pixel": int(area),
                "is_peak": is_peak,
                "is_abnormal": is_abnormal
            })
            is_abnormal_list.append(is_abnormal)

        return spot_list, is_abnormal_list

    def _draw_contours(self, img_gray: np.ndarray, contours: List[np.ndarray],
                       is_abnormal_list: List[bool]) -> np.ndarray:
        """
        在图像上绘制异常白点轮廓，异常红色，正常绿色
        """
        # 转三通道彩色图
        if len(img_gray.shape) == 2:
            img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            img_draw = img_gray.copy()

        for contour, is_abnormal in zip(contours, is_abnormal_list):
            color = self.config["abnormal_color"] if is_abnormal else self.config["normal_color"]
            cv2.drawContours(img_draw, [contour], -1, color, self.config["contour_thickness"])

            # 标注面积
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            cv2.putText(
                img_draw, f"{int(area)}px",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1
            )

        return img_draw

    def detect_single_image(self, img_path: str, save_result: bool = True, output_dir: str = "./result") -> Dict:
        """
        检测单张图像的白点
        返回：检测结果字典，包含白点列表、异常数、标注图
        """
        # 1. 读取图像
        img_raw = self._read_image(img_path)
        img_name = os.path.basename(img_path)

        # 2. 预处理
        img_pre = self._preprocess(img_raw)

        # 3. 自适应阈值分割，提取白点
        binary = cv2.adaptiveThreshold(
            img_pre, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config["adaptive_block"],
            self.config["adaptive_c"]
        )

        # 4. 形态学闭运算填充白点小孔；开运算去小噪声
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel)

        # 5. 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels <= 1:  # 只有背景
            return {
                "img_name": img_name,
                "total_spots": 0,
                "abnormal_spots": 0,
                "spot_list": [],
                "result_img": None
            }

        # 6. 局部峰值校验
        peak_flags = self._local_peak_check(img_pre, centroids[1:])

        # 7. 分类正常/异常
        spot_list, is_abnormal_list = self._classify_spots(stats, centroids, peak_flags)

        # 8. 提取轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 9. 绘制标注
        result_img = self._draw_contours(img_pre, contours, is_abnormal_list)

        # 10. 保存结果
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"result_{img_name}")
            cv2.imwrite(save_path, result_img)

        abnormal_count = sum(1 for s in spot_list if s["is_abnormal"])

        return {
            "img_name": img_name,
            "total_spots": len(spot_list),
            "abnormal_spots": abnormal_count,
            "spot_list": spot_list,
            "result_img": result_img
        }

    def calc_white_gray_mean(self, fts_path: str) -> float:
        """
        计算fts文件整张图中白点区域的灰度平均值
        """
        img_raw = self._read_image(fts_path)
        img_pre = self._preprocess(img_raw)

        # 分割白点区域
        binary = cv2.adaptiveThreshold(
            img_pre, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config["adaptive_block"],
            self.config["adaptive_c"]
        )

        # 生成白点掩码
        mask = binary > 0
        if mask.sum() == 0:
            return 0.0

        # 计算原始图像白点区域的灰度均值，保留原始位深精度
        gray_mean = np.mean(img_raw[mask])
        return round(float(gray_mean), 2)

    def batch_detect(self, img_dir: str, fts_dir: str = None, output_dir: str = "./result") -> pd.DataFrame:
        """
        批量检测指定目录下的所有图片
        :param img_dir: 待检测图片目录
        :param fts_dir: fts原始文件目录，可选，用于计算灰度均值
        :param output_dir: 结果输出目录
        :return: 检测结果DataFrame
        """
        os.makedirs(output_dir, exist_ok=True)

        # 支持的图像后缀
        support_suffix = ('.png', '.bmp', '.tiff', '.tif', '.fts', '.jpg')
        img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(support_suffix)]

        all_results = []

        for img_name in img_list:
            img_path = os.path.join(img_dir, img_name)
            try:
                # 检测白点
                detect_res = self.detect_single_image(img_path, save_result=True, output_dir=output_dir)

                # 计算fts灰度均值
                gray_mean = None
                if fts_dir:
                    fts_path = os.path.join(fts_dir, img_name)
                    if os.path.exists(fts_path):
                        gray_mean = self.calc_white_gray_mean(fts_path)

                # 整理结果
                res_row = {
                    "image_name": img_name,
                    "total_white_spots": detect_res["total_spots"],
                    "abnormal_spots": detect_res["abnormal_spots"],
                    "white_gray_mean": gray_mean
                }
                all_results.append(res_row)

                print(
                    f"处理完成: {img_name} | 白点总数: {detect_res['total_spots']} | 异常数: {detect_res['abnormal_spots']}")

            except Exception as e:
                print(f"处理失败: {img_name} | 错误: {str(e)}")
                continue

        # 生成总报表
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, "detection_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n批量检测完成，汇总报表已保存至: {csv_path}")

        return df


# ===================== 统一入口 =====================
if __name__ == "__main__":
    # 配置参数，可根据实际工艺调整
    config = {
        "min_area": 3,
        "max_area": 800,
        "circularity_threshold": 0.65,
        "area_std_factor": 2.0
    }

    # 初始化检测器
    detector = WaferWhiteSpotDetector(config)

    # 批量检测
    img_directory = "D:\E\data"  # 待检测图片目录
    # fts_directory = "./wafer_fts"  # 原始fts文件目录，没有则设为None
    output_directory = "./wafer_result"  # 结果输出目录

    result_df = detector.batch_detect(
        img_dir=img_directory,
        # fts_dir=fts_directory,
        output_dir=output_directory
    )

    # 打印整体统计
    print("\n===== 整体统计 =====")
    print(f"处理图片总数: {len(result_df)}")
    print(f"平均每图白点数量: {result_df['total_white_spots'].mean():.1f}")
    print(f"平均每图异常数量: {result_df['abnormal_spots'].mean():.1f}")
    if 'white_gray_mean' in result_df.columns and result_df['white_gray_mean'].notna().any():
        print(f"整体白点灰度均值: {result_df['white_gray_mean'].mean():.2f}")
