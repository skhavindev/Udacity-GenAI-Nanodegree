import streamlit as st
import torch
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
from PIL import Image, ImageDraw
import io
import random
from streamlit_image_coordinates import streamlit_image_coordinates


@st.cache_resource
def load_sam_model():
    model = SamModel.from_pretrained("ZhuiyiTechnology/SAM-ViT-B").eval()
    processor = SamProcessor.from_pretrained("ZhuiyiTechnology/SAM-ViT-B")
    if torch.cuda.is_available():
        model.to("cuda")
    return model, processor


@st.cache_resource
def load_inpainting_model():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        pipe.enable_sequential_cpu_offload()
    return pipe


def create_simple_mask(image, points, radius=50):
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)

    for point in points:
        x, y = point[0], point[1]
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

    return mask


def call_sam_api(image, points):
    try:
        model, processor = load_sam_model()

        inputs = processor(image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        input_points = torch.tensor([points], dtype=torch.float32)
        input_labels = torch.tensor([[1] * len(points)], dtype=torch.int32)

        if torch.cuda.is_available():
            input_points = input_points.to("cuda")
            input_labels = input_labels.to("cuda")

        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"],
                input_points=input_points,
                input_labels=input_labels,
                multimask_output=False
            )

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        mask = masks[0][0][0].numpy()
        mask = (mask > 0).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask, mode='L')

        return mask_image

    except Exception as e:
        st.warning(f"SAM inference error: {str(e)}. Using simple mask fallback.")
        return create_simple_mask(image, points)


def call_inpainting_api(image, mask, prompt, negative_prompt="", guidance_scale=7.5, seed=None):
    try:
        pipe = load_inpainting_model()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        mask = mask.resize(image.size).convert("L")

        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=20,
            strength=0.99
        ).images[0]

        return result

    except Exception as e:
        st.error(f"Inpainting error: {str(e)}")
        return None


def create_colored_mask_overlay(image, mask, subject_color=(0, 0, 255), background_color=(255, 255, 0), alpha=0.6):
    if mask.mode != 'L':
        mask = mask.convert('L')

    colored_mask = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    mask_array = np.array(mask)
    colored_array = np.zeros((mask.size[1], mask.size[0], 4), dtype=np.uint8)

    subject_mask = mask_array > 128
    colored_array[subject_mask] = [subject_color[0], subject_color[1], subject_color[2], int(255 * alpha)]

    background_mask = mask_array <= 128
    colored_array[background_mask] = [background_color[0], background_color[1], background_color[2], int(255 * alpha)]

    colored_mask = Image.fromarray(colored_array, 'RGBA')

    image_rgba = image.convert('RGBA')
    result = Image.alpha_composite(image_rgba, colored_mask)

    return result.convert('RGB')


def invert_mask(mask):
    mask_array = np.array(mask.convert('L'))
    inverted_array = 255 - mask_array
    return Image.fromarray(inverted_array, mode='L')


def add_point_to_image(image, points, point_radius=8):
    img_with_points = image.copy()
    draw = ImageDraw.Draw(img_with_points)

    for i, point in enumerate(points):
        x, y = point[0], point[1]
        draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                     fill='white', outline='green', width=3)
        draw.line([x - 3, y, x + 3, y], fill='green', width=2)
        draw.line([x, y - 3, x, y + 3], fill='green', width=2)

    return img_with_points


def main():
    st.set_page_config(page_title="Image Inpainting", layout="wide")

    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0 0.5rem 0;
    }
    .panel-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .button-container {
        display: flex;
        gap: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">Image Inpainting</div>', unsafe_allow_html=True)

    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'click_points' not in st.session_state:
        st.session_state.click_points = []
    if 'current_mask' not in st.session_state:
        st.session_state.current_mask = None
    if 'colored_mask_overlay' not in st.session_state:
        st.session_state.colored_mask_overlay = None
    if 'invert_mask_mode' not in st.session_state:
        st.session_state.invert_mask_mode = False
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = random.randint(1000000, 9999999)

    st.markdown('<div class="step-header">1. Upload an image by clicking on the first canvas.</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'],
                                     help="Upload a JPG, PNG, or JPEG image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.session_state.uploaded_image = image

        display_width = 512
        if image.width > display_width:
            ratio = display_width / image.width
            display_height = int(image.height * ratio)
            display_image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        else:
            display_image = image.copy()

        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

        with col1:
            st.markdown("📷 Input")

            if st.session_state.click_points:
                scale_ratio = display_image.width / image.width
                display_points = []
                for point in st.session_state.click_points:
                    display_x = int(point[0] * scale_ratio)
                    display_y = int(point[1] * scale_ratio)
                    display_points.append([display_x, display_y])

                img_with_points = add_point_to_image(display_image, display_points)
                coords = streamlit_image_coordinates(img_with_points, key="input_image")
            else:
                coords = streamlit_image_coordinates(display_image, key="input_image")

            if coords is not None:
                scale_ratio = image.width / display_image.width
                orig_x = int(coords["x"] * scale_ratio)
                orig_y = int(coords["y"] * scale_ratio)

                add_point = True
                for existing_point in st.session_state.click_points:
                    distance = ((orig_x - existing_point[0]) ** 2 + (orig_y - existing_point[1]) ** 2) ** 0.5
                    if distance < 20:
                        add_point = False
                        break

                if add_point:
                    st.session_state.click_points.append([orig_x, orig_y])
                    st.rerun()

        with col2:
            st.markdown("🎯 SAM result")

            if st.session_state.colored_mask_overlay is not None:
                if st.session_state.colored_mask_overlay.width > display_width:
                    ratio = display_width / st.session_state.colored_mask_overlay.width
                    display_height = int(st.session_state.colored_mask_overlay.height * ratio)
                    display_mask_overlay = st.session_state.colored_mask_overlay.resize((display_width, display_height))
                else:
                    display_mask_overlay = st.session_state.colored_mask_overlay

                st.image(display_mask_overlay, use_column_width=True)

                st.markdown("""
                <div style="display: flex; gap: 15px; margin-top: 5px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background-color: #FFD700; margin-right: 5px; opacity: 0.6;"></div>
                        <span>background</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background-color: #0000FF; margin-right: 5px; opacity: 0.6;"></div>
                        <span>subject</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Click on the input image to generate segmentation mask")

        with col3:
            st.markdown("🎨 Output")
            if "result_image" in st.session_state and st.session_state.result_image is not None:
                result_display = st.session_state.result_image.copy()
                if result_display.width > display_width:
                    ratio = display_width / result_display.width
                    display_height = int(result_display.height * ratio)
                    result_display = result_display.resize((display_width, display_height))

                st.image(result_display, use_column_width=True)

                img_buffer = io.BytesIO()
                st.session_state.result_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                st.download_button(
                    label="💾 Download Result",
                    data=img_buffer.getvalue(),
                    file_name="inpainted_result.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("Generated image will appear here")

        st.markdown(
            '<div class="step-header">2. Click on the subject you would like to keep. Immediately SAM will be run and you will see the results. If you are happy with those results move to the next step, otherwise add more points to refine your mask. Typically, the more points you add the better the segmentation.</div>',
            unsafe_allow_html=True)

        if st.session_state.click_points and (st.session_state.current_mask is None or
                                              len(st.session_state.click_points) != getattr(st.session_state,
                                                                                            'last_point_count', 0)):
            with st.spinner("🔄 Running SAM segmentation..."):
                mask = call_sam_api(image, st.session_state.click_points)

                if mask is not None:
                    st.session_state.current_mask = mask
                    st.session_state.colored_mask_overlay = create_colored_mask_overlay(image, mask)
                    st.session_state.last_point_count = len(st.session_state.click_points)
                    st.rerun()
                else:
                    st.error("Failed to generate mask. Please try again.")

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.click_points = []
                st.session_state.current_mask = None
                st.session_state.colored_mask_overlay = None
                if "result_image" in st.session_state:
                    del st.session_state.result_image
                st.rerun()

        if st.session_state.current_mask is not None:
            st.markdown(
                '<div class="step-header">3. Write a prompt (and optionally a negative prompt) for what you want to generate for the infilling. Adjust the CFG scale and the seed if needed. You can also invert the mask, i.e., infill the subject instead of the background by toggling the relative checkbox.</div>',
                unsafe_allow_html=True)

            st.markdown("Parameters")

            col_param1, col_param2, col_param3 = st.columns([2, 2, 1])

            with col_param1:
                guidance_scale = st.slider("Classifier-Free Guidance Scale",
                                           min_value=1.0, max_value=20.0, value=7.0, step=0.5)

            with col_param2:
                st.session_state.random_seed = st.number_input("Random seed",
                                                               value=st.session_state.random_seed,
                                                               min_value=1000000, max_value=9999999)

            with col_param3:
                st.session_state.invert_mask_mode = st.checkbox("Infill subject instead of background",
                                                                value=st.session_state.invert_mask_mode)

            col_prompt1, col_prompt2 = st.columns([1, 1])

            with col_prompt1:
                prompt = st.text_area("Prompt for infill",
                                      placeholder="e.g., 'a beautiful mountain landscape'",
                                      height=100)

            with col_prompt2:
                negative_prompt = st.text_area("Negative prompt",
                                               placeholder="e.g., 'blurry, low quality'",
                                               height=100)

            col_action1, col_action2 = st.columns([1, 1])

            with col_action2:
                if st.button("🎨 Run Inpaint", use_container_width=True, type="primary"):
                    if prompt.strip():
                        with st.spinner("🎨 Generating inpainted image... This may take a minute..."):
                            if st.session_state.invert_mask_mode:
                                final_mask = st.session_state.current_mask
                            else:
                                final_mask = invert_mask(st.session_state.current_mask)

                            result_image = call_inpainting_api(
                                image,
                                final_mask,
                                prompt,
                                negative_prompt,
                                guidance_scale,
                                st.session_state.random_seed
                            )

                            if result_image is not None:
                                st.session_state.result_image = result_image
                                st.success("✅ Inpainting completed!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to generate inpainted image. Please try again.")
                    else:
                        st.warning("⚠️ Please enter a prompt before running inpainting.")

    else:
        st.info("👆 Please upload an image to get started")

    with st.expander("💡 Examples"):
        st.markdown("""
        Example prompts:
        - Background replacement: "a cat in the park", "a person in space", "beautiful sunset over mountains"  
        - Subject replacement: "a red sports car", "a golden retriever", "a crocodile"

        Tips:
        - Click multiple times on different parts of the object for better segmentation
        - Use detailed, descriptive prompts for better results
        - Higher guidance scale values (10-15) follow prompts more closely
        - Lower values (5-7) allow more creative interpretation
        """)

    with st.expander("⚙️ Model Information"):
        st.markdown("""
        Models: Lightweight SAM (ZhuiyiTechnology/SAM-ViT-B) + Stable Diffusion Inpainting (stabilityai/stable-diffusion-2-inpainting)

        Requirements: This app runs models locally. Ensure you have PyTorch, Transformers, and Diffusers installed. GPU recommended for speed.
        """)


if __name__ == "__main__":
    main()
