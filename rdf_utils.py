import py4DSTEM
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def analyze_image_rdf_pdf(image_path, crop_size=100, pixel_size=0.00625, q_range=(50, 72), center_guess=(157, 172), density=0.01):
    """
    分析单张图片的径向分布函数（RDF）和对分布函数（PDF）。

    参数：
        image_path (str): 图片路径。
        crop_size (int, optional): 裁剪大小。默认为100。
        pixel_size (float, optional): 像素大小（Å^-1）。默认为0.00641。
        q_range (tuple, optional): 椭圆拟合的径向范围。默认为(50, 72)。
        center_guess (tuple, optional): 椭圆中心初始猜测值。默认为(157, 172)。
        density (float, optional): 样品密度。默认为0.01。

    返回：
        tuple: 包含径向距离、径向分布函数和对分布函数的元组。
    """
    # TODO: 拟合椭圆
    # TODO: 极坐标变换

def analyze_image_rdf_pdf(image_path, crop_size=30, pixel_size=0.00641, q_range=(50, 72), center_guess=(157, 172), density=0.01):
    """
    分析单张图片的径向分布函数（RDF）和对分布函数（PDF）。

    参数：
        image_path (str): 图片路径。
        crop_size (int, optional): 裁剪大小。默认为100。
        pixel_size (float, optional): 像素大小（Å^-1）。默认为0.00641。
        q_range (tuple, optional): 椭圆拟合的径向范围。默认为(50, 72)。
        center_guess (tuple, optional): 椭圆中心初始猜测值。默认为(157, 172)。
        density (float, optional): 样品密度。默认为0.01。

    返回：
        tuple: 包含径向距离、径向分布函数、对分布函数和椭圆拟合参数的元组。
    """

    # 加载图像并转换为灰度
    img = Image.open(image_path).convert('L')
    arr = np.array(img)

    # 裁剪图像
    arr = arr[crop_size:-crop_size, crop_size:-crop_size]
    datacube = py4DSTEM.DataCube(arr[None, None, :, :])

    # 设置校准参数
    datacube.calibration.set_Q_pixel_size(pixel_size)
    datacube.calibration.set_Q_pixel_units('A^-1')

    # 计算 dp_mean
    datacube.get_dp_mean()

    print(datacube.tree('dp_mean'))

    # 椭圆拟合（不使用掩膜）
    params = py4DSTEM.process.calibration.fit_ellipse_1D(
        datacube.tree('dp_mean'),
        center=center_guess,
        fitradii=q_range,
    )

    # 应用椭圆校准
    datacube.calibration.set_origin(params[:2])
    datacube.calibration.set_p_ellipse(params[:5])

    # 极坐标转换（不使用掩膜）
    polar_datacube = py4DSTEM.PolarDatacube(
        datacube,
        qmin=0.0,
        qmax=170.0,
        qstep=1.0,
        n_annular=90,
        two_fold_symmetry=True,
        qscale=1.0,
    )

    polar_datacube.calculate_radial_statistics()

    # 计算径向分布函数
    radial_distance, g_r, pdf = polar_datacube.calculate_pair_dist_function(
        k_min=0.25,
        k_width=0.10,
        damp_origin_fluctuations=True,
        density=density,
        returnval=True,
    )

    return radial_distance, g_r, pdf, params, datacube


def get_center(data, thres=0.5):
    """
    根据阈值计算图像的质心作为中心点。

    参数：
        data (numpy.ndarray): 输入图像数据。
        thres (float, optional): 阈值比例，用于确定哪些像素点参与质心计算。默认为 0.5。

    返回：
        tuple: 图像质心坐标 (x, y)。
    """
    data_max = np.max(data)
    data_min = np.min(data)
    data_thres = (data_max - data_min) * thres + data_min
    data_binary = (data >= data_thres).astype(np.float32)  # 二值化图像
    mass = np.sum(data_binary)
    center_x = np.sum(data_binary, axis=0) @ np.arange(data.shape[1]) / mass
    center_y = np.sum(data_binary, axis=1) @ np.arange(data.shape[0]) / mass
    return center_x, center_y