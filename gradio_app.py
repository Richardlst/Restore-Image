import os
import sys
import cv2
import numpy as np
import logging
import gradio as gr
from PIL import Image
import io
import base64
import torch
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gradio_app')

# Thêm đường dẫn đến thư mục Inpainting
inpainting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Inpainting')
sys.path.append(inpainting_dir)

# Import các chức năng từ các module hiện có
# Import chức năng nâng cao từ enhance.py
try:
    from enhance import super_resolution
    logger.info("Tải thành công enhance.py")
except ImportError as e:
    logger.error(f"Không thể import từ enhance.py: {e}")
    def super_resolution(image, noise_level=0, noise_reduction=False):
        return image, "Chức năng Super Resolution không khả dụng"

# Import chức năng tô màu từ colorize.py  
try:
    from colorize import colorize_image
    logger.info("Tải thành công colorize.py")
except ImportError as e:
    logger.error(f"Không thể import từ colorize.py: {e}")
    def colorize_image(image):
        return image, "Chức năng Colorization không khả dụng"

# Import trực tiếp cả mô-đun test.py
try:
    sys.path.append(inpainting_dir)  # Thêm đường dẫn Inpainting vào sys.path
    import test  # Import cả module để mô hình được tải nằm trong test
    from test import process, generate_seed  # Chỉ import các hàm cần thiết
    logger.info("Tải thành công các hàm từ test.py")
    stable_diffusion_available = True
except ImportError as e:
    logger.error(f"Không thể import từ test.py: {e}")
    stable_diffusion_available = False

# Thiết lập môi trường
os.makedirs("cache", exist_ok=True)
os.makedirs("result", exist_ok=True)

# Xóa bỏ các biến và hàm load_models không cần thiết 
# vì chúng ta sử dụng trực tiếp từ test.py

# Các hàm xử lý
# Xóa bỏ hàm process_inpainting vì chúng ta sử dụng trực tiếp process từ test.py
# Giải phóng bộ nhớ và đánh lừa GPU hơn

def process_super_resolution(image, noise_level=0, noise_reduction=False):
    """
    Xử lý nâng cao chất lượng ảnh với tùy chọn lọc nhiễu
    """
    try:
        if image is None:
            return None, "Vui lòng tải lên ảnh trước khi xử lý"
        
        # Chuyển đổi đầu vào thành numpy array
        img_np = np.array(image)
        
        # Đảm bảo ảnh là RGB
        if len(img_np.shape) == 2:  # Grayscale
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Áp dụng lọc nhiễu nếu được chọn
        if noise_reduction:
            logger.info(f"Áp dụng lọc nhiễu với mức: {noise_level}")
            if noise_level > 15:
                # Lọc nhiễu tăng cường cho nhiễu nặng
                # Kết hợp Non-Local Means và lọc median thích ứng
                img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
                
                # Phát hiện cạnh để bảo toàn chi tiết
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Lọc median thích ứng
                mask = edges == 0  # Không áp dụng lọc cho các cạnh
                temp = img_np.copy()
                temp[mask] = cv2.medianBlur(img_np, 5)[mask]
                img_np = temp
            elif noise_level > 5:
                # Lọc nhiễu vừa phải
                img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 5, 5, 7, 21)
                
                # Lọc song phương để bảo toàn cạnh
                img_np = cv2.bilateralFilter(img_np, 9, 75, 75)
            else:
                # Lọc nhiễu nhẹ
                img_np = cv2.bilateralFilter(img_np, 5, 50, 50)
        
        # Áp dụng super-resolution
        logger.info("Áp dụng super-resolution")
        enhanced_img = super_resolution(img_np)
        
        return enhanced_img, "Nâng cao chất lượng ảnh thành công"
    except Exception as e:
        logger.error(f"Lỗi trong quá trình nâng cao chất lượng: {e}")
        return None, f"Lỗi: {str(e)}"

def process_colorization(image):
    """
    Tô màu ảnh đen trắng
    """
    try:
        if image is None:
            return None, "Vui lòng tải lên ảnh trước khi xử lý"
        
        # Chuyển đổi đầu vào thành numpy array
        img_np = np.array(image)
        
        # Xử lý định dạng ảnh
        if len(img_np.shape) == 2:  # Grayscale
            # Chuyển grayscale thành BGR cho colorize_image
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif len(img_np.shape) == 3:  # RGB/RGBA
            if img_np.shape[2] == 4:  # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            # Chuyển RGB sang BGR cho colorize_image
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Áp dụng tô màu (hàm mong đợi đầu vào là BGR format)
        logger.info("Áp dụng tô màu")
        colorized_img = colorize_image(img_np, render_factor=35)  # Kết quả trả về là BGR
        
        # Chuyển lại BGR sang RGB để hiển thị trong Gradio
        colorized_img = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)
        
        return colorized_img, "Tô màu ảnh thành công"
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tô màu: {e}")
        return None, f"Lỗi: {str(e)}"

# Xây dựng giao diện Gradio
# CSS nổi bật và đẹp mắt
css = """
/* Reset CSS cho background */
#root, #component-0, .gradio-container, .main, body, html {
    background: linear-gradient(135deg, #89f7fe, #66a6ff) !important;
    background-color: #89f7fe !important;
}

/* Bo tròn tất cả các khung ảnh và input */
.gr-image-container, .gr-image, .gr-input, .gr-box, .gr-panel, .gr-image-tool, 
img, .gr-card, .gradio-box, .output-image, .input-image, .gr-accordion {
    border-radius: 20px !important;
    overflow: hidden !important;
}

/* Bo tròn các khung output */
.output-image, .output-markdown, .output-html, .output-text, 
.gr-panel.output, .gr-box, .gr-panel, .gr-input-label, .preview-image {
    border-radius: 20px !important;
    overflow: hidden !important;
}

/* Thêm viền cho các khung ảnh */
.gr-image-container, .gr-image, img.svelte-17ttdjy, .gr-panel.output {
    border: 2px solid rgba(79, 172, 254, 0.5) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

.contain {
    background: linear-gradient(135deg, #89f7fe, #66a6ff) !important;
    background-color: #89f7fe !important;
}

footer {
    display: none !important;
}

/* Giao diện nổi bật và đẹp mắt - Nền xanh nhạt để làm nổi bật nút màu xanh đậm (#4facfe) */
body.gradio-container {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #accbee, #e7f0fd) !important;
    color: #333;
    position: relative;
    z-index: 0;
    overflow: hidden;
}

/* Thêm pattern cho background */
body.gradio-container:before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
        radial-gradient(circle at 25% 25%, rgba(172, 203, 238, 0.6) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(231, 240, 253, 0.4) 0%, transparent 50%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%234facfe' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    z-index: -1;
    opacity: 0.8;
}

/* Hiệu ứng shine cho nún bấm */
@keyframes shine {
    0% {
        background-position: -100% 0;
    }
    100% {
        background-position: 200% 0;
    }
}

/* Header */
.app-header h1 {
    color: white !important;
    font-size: 2.4rem;
    margin: 0 0 0.5rem 0;
    text-align: center;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.app-header p {
    color: #4a5568;
    text-align: center;
    margin-bottom: 1.5rem;
    font-size: 1.2rem;
}

/* Tabs */
.tabs {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    
    border: 1px solid rgba(79, 172, 254, 0.3);
}

.tab-nav {
    background: rgba(255, 255, 255, 0.6);
    border-bottom: 1px solid rgba(79, 172, 254, 0.3);
    padding: 0.8rem;
    display: flex;
    justify-content: center;
}

.tab-nav button {
    padding: 0.8rem 1.5rem;
    margin: 0 0.5rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1.1rem;
    border: none;
    background: rgba(167, 199, 237, 0.6);
    color: #2a4365;
    transition: all 0.3s;
    border: 1px solid rgba(79, 172, 254, 0.2);
}

.tab-nav button[data-selected="true"] {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    color: white;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.5);
}

.tab-content {
    padding: 1.5rem;
    background: rgba(0, 0, 0, 0.1);
}

/* Image editor */
.dark-image-editor {
    background-color: rgba(255, 255, 255, 0.8) !important;
    border: 2px solid rgba(79, 172, 254, 0.5) !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    height: 100%;
    min-height: 512px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15) !important;
}

.dark-image-editor .toolbar {
    background: rgba(231, 240, 253, 0.9) !important;
    border-bottom: 1px solid rgba(79, 172, 254, 0.3) !important;
    padding: 10px;
}

.dark-image-editor canvas {
    background: rgba(255, 255, 255, 0.7) !important;
}

/* Buttons */
button.primary-btn {
    background: linear-gradient(135deg, #4facfe, #00f2fe) !important;
    color: white !important;
    padding: 0.8rem 1.5rem !important;
    border-radius: 15px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.3s !important;
    width: auto !important;
    display: inline-block !important;
    margin: 0.8rem 0 !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    position: relative !important;
    overflow: hidden !important;
    z-index: 1 !important;
    min-width: 220px !important;
    font-size: 1.1rem !important;
}

button.primary-btn:before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 200% !important;
    height: 100% !important;
    background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
    animation: shine 2s infinite !important;
    z-index: -1 !important;
}

button.primary-btn:hover {
    background: linear-gradient(135deg, #00f2fe, #4facfe) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3) !important;
}

.random-btn {
    background: linear-gradient(135deg, #f6d365, #fda085) !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-size: 1.3rem !important;
    width: auto !important;
    height: auto !important;
    padding: 0.5rem 1rem !important;
    margin-left: 0.8rem !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s !important;
}

.random-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3) !important;
}

/* Input elements */


/* Slider styling */
input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    height: 8px !important;
    background: rgba(255, 255, 255, 0.4) !important;
    border-radius: 4px !important;
    outline: none !important;
    margin: 1rem 0 !important;
    max-width: 100% !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Container cho sliders */
:root {
    --border-color-primary: #ffffff !important;
}
.slider-container {
    padding: 0 !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    width: 100% !important;
    height: 200px !important;
    max-width: 100% !important;
    overflow: hidden !important;
}
.status-container {
    height: 200px !important;
}

/* Ẩn tất cả các thanh cuộn */
*::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

* {
    scrollbar-width: none !important; /* Firefox */
    -ms-overflow-style: none !important; /* IE and Edge */
}

/* Đảm bảo các container vẫn có thể cuộn nhưng không hiển thị thanh cuộn */
.gradio-container, .gradio-box, .gr-panel, .gr-box, body, html, .gr-container, .gr-form {
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 20px !important;
    height: 20px !important;
    background: linear-gradient(135deg, #4facfe, #00f2fe) !important;
    border-radius: 50% !important;
    cursor: pointer !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3) !important;
}

/* Checkbox styling */
input[type="checkbox"] {
    accent-color: #4facfe !important;
    width: 18px !important;
    height: 18px !important;
}

/* Layout fixes */
.container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 1rem !important;
}

.row {
    display: flex !important;
    flex-wrap: wrap !important;
    margin: 0 -0.5rem !important;
}

.col {
    padding: 0 0.5rem !important;
    flex: 1 !important;
    min-width: 300px !important;
}

/* Ensure consistent spacing */
.block {
    margin-bottom: 1.5rem !important;
}

/* Fix image display */
img {
    max-width: 100% !important;
    height: auto !important;
}
"""

# Tạo brush cho ImageEditor với độ mờ cao hơn để dễ nhìn hơn
brush = gr.Brush(
    colors=["rgba(255, 0, 0, 0.8)"],  # Chỉ dùng màu đỏ với độ mờ cao hơn
    color_mode="fixed"
)

with gr.Blocks(
        title="Image Restoration Tool", 
        css=css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
            neutral_hue="blue",
            radius_size=gr.themes.sizes.radius_xxl, # Tăng độ bo tròn cho tất cả thành phần
            font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"]
        ).set(button_primary_background_fill="linear-gradient(135deg, #4facfe, #00f2fe)"),
    ) as demo:
    with gr.Column(elem_classes="app-header"):
        gr.Markdown("# Image Restoration Tool")
        gr.Markdown("Công cụ phục hồi & cải thiện chất lượng ảnh bằng AI")
    
    with gr.Tabs(elem_classes="tabs"):
        # Tab Inpainting
        with gr.Tab("✨ Image Inpainting"):
            gr.Markdown("## 🎨 Inpainting Image")
            
            with gr.Row():
                with gr.Column():
                    # Sử dụng ImageEditor với layers=True để bật tính năng hiển thị real-time
                    editor = gr.ImageEditor(
                        type="pil",
                        label="Dùng brush để đánh dấu vùng cần chỉnh sửa",
                        brush=brush,
                        interactive=True,
                        transforms=[],
                        layers=True,  # Bật layers để hiển thị nét vẽ real-time khi di chuyển chuột
                        elem_classes="dark-image-editor",
                        height=512,
                        width=512,
                        sources=["upload", "clipboard"]  # Chỉ cho phép upload từ thiết bị và clipboard
                    )
                with gr.Column():
                    # Sử dụng gr.Image chỉ để hiển thị kết quả, không cho phép upload
                    inpaint_output = gr.Image(
                        label="Kết quả", 
                        height=512, 
                        type="pil", 
                        interactive=False,     # Cho phép tương tác với ảnh (phóng to, thu nhỏ)
                        show_download_button=True,  # Hiển thị nút tải xuống
                        show_share_button=True,     # Hiển thị nút chia sẻ
                        show_label=True,     # Hiển thị nhãn
                        container=True,      # Đảm bảo hiển thị trong container rõ ràng
                        sources=[],          # Không cho phép upload (danh sách rỗng)
                        elem_id="inpaint_output_image"  # ID duy nhất để CSS đặc biệt
                    )
            
            # Prompt ẩn và các tùy chọn
            if stable_diffusion_available:
                # Sử dụng hidden textbox cho prompt để không hiển thị trên giao diện
                prompt = gr.Textbox(visible=False, value="Restore and enhance the image naturally")
                
                with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                    with gr.Row():
                        with gr.Column():
                            diffusion_size = gr.Slider(8, 1024, value=512, step=8, label="Diffusion Size")
                            output_size = gr.Slider(8, 4096, value=1024, step=1, label="Image Output Size")
                        with gr.Column():
                            sampling_step = gr.Slider(1, 250, value=50, step=1, label="Sampling Step")
                            guidance_scale = gr.Slider(1, 10, value=7.5, step=0.1, label="Guidance Scale") 
                    with gr.Row():
                        strength = gr.Slider(0, 1, value=0.8, step=0.1, label="Denoising Strength")
                    with gr.Row():
                        seed = gr.Textbox(label="Seed", value=generate_seed(), scale=10)
                        random_btn = gr.Button("🎲", scale=1, elem_classes="random-btn")
                
                # Liên kết nút tạo seed ngẫu nhiên
                random_btn.click(generate_seed, None, seed)
                
                # Nút thực hiện inpainting
                run_btn = gr.Button("✨ Thực hiện Inpainting", variant="primary", elem_classes="primary-btn")
            
            # Sử dụng wrapper function để đảm bảo kết quả được hiển thị chính xác
            def process_wrapper(editor_data, prompt, diffusion_size, output_size, guidance_scale, sampling_step, strength, seed):
                try:
                    # Lấy kích thước gốc của ảnh đầu vào
                    original_img = editor_data["background"]
                    original_size = original_img.size
                    print(f"[DEBUG] Kích thước ảnh gốc: {original_size}")
                    
                    # Đặt output_size là kích thước tối đa của ảnh gốc để giữ nguyên tỷ lệ
                    output_size = max(original_size)
                    print(f"[DEBUG] Đặt kích thước đầu ra = {output_size} để giữ nguyên tỷ lệ")
                    
                    # Gọi hàm process từ test.py
                    result = process(editor_data, prompt, diffusion_size, output_size, guidance_scale, sampling_step, strength, seed)
                    
                    # Đảm bảo kết quả có kích thước giống ảnh gốc
                    if result is not None and isinstance(result, Image.Image):
                        # Nếu kích thước khác, resize về đúng kích thước gốc
                        if result.size != original_size:
                            print(f"[DEBUG] Resizing từ {result.size} về {original_size}")
                            result = result.resize(original_size, Image.LANCZOS)
                        
                        # Lưu lại kết quả
                        os.makedirs("./result", exist_ok=True)
                        result.save('./result/inpainted_output.png')
                        return result
                    
                    # Nếu có lỗi trong quá trình xử lý, thử đọc từ file đã lưu
                    if os.path.exists('./result/inpainted_output.png'):
                        img = Image.open('./result/inpainted_output.png')
                        # Resize về đúng kích thước gốc nếu cần
                        if img.size != original_size:
                            img = img.resize(original_size, Image.LANCZOS)
                        return img
                        
                    return result
                    
                except Exception as e:
                    print(f"Lỗi trong quá trình xử lý: {e}")
                    # Vẫn thử đọc ảnh đã lưu nếu có
                    if os.path.exists('./result/inpainted_output.png'):
                        try:
                            return Image.open('./result/inpainted_output.png')
                        except:
                            pass
                    return None
                    
            run_btn.click(
                process_wrapper,
                inputs=[editor, prompt, diffusion_size, output_size, guidance_scale, sampling_step, strength, seed],
                outputs=inpaint_output
            )
        
        # Tab Super Resolution
        with gr.Tab("🔎 Super Resolution"):
            with gr.Row():
                sr_input = gr.Image(
                    label="Ảnh gốc", 
                    height=400, 
                    type="pil", 
                    image_mode="RGB",           # Đảm bảo định dạng ảnh nhất quán
                    show_label=True,           # Hiển thị nhãn rõ ràng hơn
                    show_download_button=True,  # Cho phép tải xuống
                    sources=["upload", "clipboard"]  # Chỉ cho phép upload từ thiết bị và clipboard
                )
            
                sr_output = gr.Image(
                    label="Kết quả", 
                    height=400, 
                    type="pil",                # Sử dụng type="pil" thay vì "numpy" 
                    interactive=False,          # Phóng to thu nhỏ
                    show_download_button=True, # Nút tải xuống
                    show_label=True,          # Hiển thị nhãn
                    container=True,            # Đảm bảo hiển thị trong container rõ ràng
                    sources=[],                # Không cho phép upload (danh sách rỗng)
                    elem_id="sr_output_image"  # ID duy nhất để CSS đặc biệt
                )
            with gr.Row():
                with gr.Group(elem_classes="slider-container"):
                    sr_noise_level = gr.Slider(label="Độ nét (0-20)", minimum=0, maximum=20, value=0, step=1)
                    sr_noise_reduction = gr.Checkbox(label="Áp dụng giảm nhiễu nâng cao", value=False)
                with gr.Group(elem_classes="status-container"):
                    sr_status = gr.Textbox(label="Trạng thái", value="Chưa xử lý", interactive=False)
            with gr.Row():
                sr_btn = gr.Button("Xử lý Super Resolution", variant="primary", elem_classes="primary-btn")
        
            # Liên kết nút xử lý
            sr_btn.click(
                fn=process_super_resolution,
                inputs=[sr_input, sr_noise_level, sr_noise_reduction],
                outputs=[sr_output, sr_status]
            )

        
        # Tab Colorization
        with gr.Tab("🌈 Colorization"):
            with gr.Row():
                with gr.Column():
                    color_input = gr.Image(
                        label="Ảnh đen trắng", 
                        height=400, 
                        type="pil",
                        image_mode="RGB",          # Đảm bảo định dạng ảnh nhất quán
                        show_label=True,           # Hiển thị nhãn rõ ràng hơn
                        show_download_button=True,  # Cho phép tải xuống
                        sources=["upload", "clipboard"]  # Chỉ cho phép upload từ thiết bị và clipboard
                    )
                    color_btn = gr.Button("Xử lý Colorization", variant="primary", elem_classes="primary-btn")
                
                with gr.Column():
                    color_output = gr.Image(
                        label="Kết quả", 
                        height=400, 
                        type="pil",                # Sử dụng type="pil" thay vì "numpy" 
                        interactive=False,          # Phóng to thu nhỏ
                        show_download_button=True, # Nút tải xuống
                        show_label=True,          # Hiển thị nhãn
                        container=True,            # Đảm bảo hiển thị trong container rõ ràng
                        sources=[],                # Không cho phép upload (danh sách rỗng)
                        elem_id="color_output_image"  # ID duy nhất để CSS đặc biệt
                    )
                    color_status = gr.Textbox(label="Trạng thái", value="Chưa xử lý", interactive=False)
            
            # Liên kết nút xử lý
            color_btn.click(
                fn=process_colorization,
                inputs=[color_input],
                outputs=[color_output, color_status]
            )
    

# Mô hình được tải khi import, giống như cách làm trong test.py
print("\nĐang tự động tải mô hình...")
if stable_diffusion_available:
    # Load mô hình inpaint từ test.py
    print("Tải mô hình Stable Diffusion Inpainting...")
    try:
        # Sử dụng trực tiếp các hàm và mô hình đã được tải từ test.py khi import
        from test import inpaint_pipe, dat_model
        print(f"Sử dụng GPU: {inpaint_pipe.device}")
    except Exception as e:
        print(f"Lỗi: {e}")

# Khởi chạy ứng dụng
if __name__ == "__main__":
    demo.launch(share=False)
