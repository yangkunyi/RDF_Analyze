import streamlit as st
import py4DSTEM
from py4DSTEM import visualize
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import periodictable as pt
from rdf_utils import analyze_image_rdf_pdf, get_center

st.title("RDF Analysis for Images")

# 文件上传
with st.sidebar:
    st.header("Parameter Selection")

    # 文件上传
    uploaded_file = st.file_uploader("Upload a image", type=["tif", "tiff", "png", "jpg", "jpeg"])

    # 元素周期表选择器
    selected_element = st.selectbox("Element", options=[None] + list(pt.elements))

    if selected_element is not None:
        density = selected_element.density / (1.67 * selected_element.mass)  # atoms/Å^3
        st.write(f"Selected Element: {selected_element.name}, Density: {density:.5f} atoms/Å^3")
    else:
        density = st.number_input("Density (atoms/Å^3)", value=0.01, format="%.5f")

    # 像素大小输入
    pixel_size = st.number_input("Pixel Size (Å^-1)", value=0.00641, format="%.5f", min_value=0.0, step=0.0002)

    # 预处理参数
    with st.expander("Preprocessing"):
        thres = st.slider("Threshold for Center Calculation", 0.0, 1.0, 0.5, 0.01)
        crop_ratio = st.slider("Crop Ratio", 0.0, 0.5, 0.2, 0.01)
        q_range_percent = st.slider("Guess Ellipse Range (%)", 0, 100, (25, 50), 1)

    # 分析参数
    with st.expander("Analyzing"):
        
        ana_q_range_percent = st.slider("Integral Q Range (%)", 0, 100, (0, 100), 1)


# # 参数设置


# center_guess = st.slider("Center Guess", 0, 300, (128, 128), 1)
# density = st.number_input("Density (atoms/Å^3)", value=0.01, format="%.5f")

if uploaded_file is not None:
    # 读取图像数据
    img = Image.open(uploaded_file).convert('L')
    # 预处理部分
    st.header("Preprocessing")

    center_x, center_y = get_center(np.array(img), thres)
    st.write(f"Calculated Center: ({center_x:.2f}, {center_y:.2f})")

    # 将图像置于中心
    img_centered = Image.fromarray(np.roll(np.array(img), (int(img.height/2 - center_y), int(img.width/2 - center_x)), axis=(0, 1)))
    st.image(img_centered, caption="Centered Image", use_column_width=True)

    # 裁剪图像
    height, width = img_centered.size  # 使用置中后的图像尺寸
    crop_size_h = int(height * crop_ratio / 2)
    crop_size_w = int(width * crop_ratio / 2)
    img_cropped = img_centered.crop((crop_size_w, crop_size_h, width - crop_size_w, height - crop_size_h))
    st.image(img_cropped, caption="Cropped Image", use_column_width=True)



    # 拟合椭圆
    arr = np.array(img_cropped)  # 使用裁剪后的图像
    datacube = py4DSTEM.DataCube(arr[None, None, :, :])
    datacube.calibration.set_Q_pixel_size(pixel_size)
    datacube.calibration.set_Q_pixel_units('A^-1')
    datacube.get_dp_mean()

    # 获取图像尺寸
    height, width = img_cropped.size

    # 计算 q_range (使用单个滑块)
    max_radius = min(height, width) / 2  # 最大半径，单位为 Å^-1
    q_range_start = q_range_percent[0] / 100 * max_radius
    q_range_end = q_range_percent[1] / 100 * max_radius
    q_range = (q_range_start, q_range_end)  # 更新 q_range

    
    #### 椭圆拟合
    # print(q_range)
    params = py4DSTEM.process.polar.fit_amorphous_ring(
        datacube.tree('dp_mean'),
        center = (width / 2, height / 2),
        radial_range = q_range, 
        verbose = False,
        figsize = (8,8)
    )

    plt.title("Ellipse Fit Result")
    st.pyplot(plt.gcf())

    params = params[:5]

    # 应用椭圆校准
    datacube.calibration.set_origin(params[:2])
    datacube.calibration.set_p_ellipse(params[:5])


    st.header("Analyzing")
    # 计算rdf

    ana_q_range_min = ana_q_range_percent[0] / 100 * max_radius
    ana_q_range_max = ana_q_range_percent[1] / 100 * max_radius

    fig2, ax2 = plt.subplots()
    visualize.show(datacube.tree('dp_mean'), figax=(fig2, ax2), ticks=False)
    circle_inner = plt.Circle((width-params[0],height-params[1]), ana_q_range_min, color='r', fill=False, linestyle='--')
    circle_outer = plt.Circle((width-params[0],height-params[1]), ana_q_range_max, color='r', fill=False, linestyle='--')
    ax2.add_patch(circle_inner)
    ax2.add_patch(circle_outer)
    ax2.set_title("Integral Q Range Visualization")
    st.pyplot(fig2)

    st.header("Analysis Results")

    
    polar_datacube = py4DSTEM.PolarDatacube(
        datacube,
        qmin=ana_q_range_min,
        qmax=ana_q_range_max,
        qstep=1.0,
        n_annular=180,
        two_fold_symmetry=True,
        qscale=1.0,
    )

    py4DSTEM.show( 
        [
            datacube.data[0,0],
            polar_datacube.data[0,0],
        ],
        cmap = 'turbo',
        ticks = False,
    )

    st.subheader("Pattern in Cartesian and Polar")
    st.pyplot(plt.gcf())

    # # 计算径向平均值
    polar_datacube.calculate_radial_statistics()

    # # 计算径向分布函数
    radial_distance, g_r, pdf = polar_datacube.calculate_pair_dist_function(
        k_min=0.25,
        k_width=0.10,
        damp_origin_fluctuations=True,
        density=density,
        returnval=True,
    )

    # 显示结果
    


# Reduced RDF 图
    st.subheader("RDF")
    fig_gr, ax_gr = plt.subplots()  # 设置图形大小
    ax_gr.plot(radial_distance, g_r)
    ax_gr.set_xlabel('Radial Distance (Å)')
    ax_gr.set_ylabel('Radial Distribution Function (RDF)')
    st.pyplot(fig_gr)

    st.subheader("PDF")
    fig_pdf, ax_pdf = plt.subplots()  # 设置图形大小
    ax_pdf.plot(radial_distance, pdf)
    ax_pdf.set_xlabel('Radial Distance (Å)')
    ax_pdf.set_ylabel('Pair Distribution Function (PDF)')
    st.pyplot(fig_pdf)
