import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.markdown("""
    <style>
        /* Keyframes for gradient animation */
        @keyframes gradient {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        
        /* Main container styling with animated gradient */
        .main {
            background: linear-gradient(
                135deg, 
                rgba(230, 205, 16, 0.9),
                rgba(55, 195, 0, 0.86),
                rgba(230, 205, 16, 0.9)
            );
            background-size: 200% 200%;
            animation: gradient 10s ease infinite;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            margin: 1rem;
        }
        
        /* Custom background with animated overlay */
        .stApp {
            background-image: 
                linear-gradient(
                    45deg,
                    rgba(0, 0, 0, 0.7),
                    rgba(0, 0, 0, 0.3),
                    rgba(0, 0, 0, 0.7)
                ),
                url('https://t3.ftcdn.net/jpg/06/19/82/00/240_F_619820028_0rkb6i8sHldgaAsDAYplQVqQPGC0fr5J.jpg');
            background-size: 200% 200%, cover;
            background-position: center;
            background-attachment: fixed;
            animation: gradient 75s ease infinite;
        }
        
        /* Animated title styling */
        .title {
            background: linear-gradient(
                45deg, 
                #f1c40f,
                #e67e22,
                #f1c40f
            );
            background-size: 200% 200%;
            animation: gradient 8s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1.5rem;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Animated radio button styling */
        .stRadio > label {
            color: #ffffff !important;
            font-size: 1.2rem;
            background: linear-gradient(
                45deg,
                rgb(1, 47, 3),
                rgb(214, 255, 31),
                rgb(1, 47, 3)
            );
            background-size: 200% 200%;
            animation: gradient 8s ease infinite;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        /* Animated button styling */
        .stButton > button {
            background: linear-gradient(
                45deg, 
                #2ecc71,
                #27ae60,
                #2ecc71
            );
            background-size: 200% 200%;
            animation: gradient 6s ease infinite;
            color: green;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            font-weight: bold;
            text-transform: uppercase;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            animation: gradient 3s ease infinite;
        }
        
        /* Animated success message styling */
        .element-container .stSuccess {
            background: linear-gradient(
                45deg, 
                rgba(46, 204, 113, 0.8),
                rgba(39, 174, 96, 0.8),
                rgba(46, 204, 113, 0.8)
            );
            background-size: 200% 200%;
            animation: gradient 8s ease infinite;
            padding: 1rem;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Animated file uploader styling */
        .uploadedFile {
            background: linear-gradient(
                45deg,
                #3498db,
                #2980b9,
                #3498db
            );
            background-size: 200% 200%;
            animation: gradient 8s ease infinite;
            padding: 1rem;
            border-radius: 10px;
            color: white;
        }
        
        /* Result container styling */
        .result-container {
            background: linear-gradient(
                45deg,
                rgba(255, 255, 255, 0.9),
                rgba(230, 230, 230, 0.9),
                rgba(255, 255, 255, 0.9)
            );
            background-size: 200% 200%;
            animation: gradient 10s ease infinite;
            padding: 1.5rem;
            border-radius: 15px;
            margin-top: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Image container styling */
        .stImage {
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .stImage:hover {
            transform: scale(1.02);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.45);
        }
    </style>
""", unsafe_allow_html=True)


category_model = load_model("C:/Users/aniru/Downloads/mangorcatgry.h5")
ripeness_model = load_model("C:/Users/aniru/Downloads/mangoripeness.keras")


categories = ["Alphonso", "Ambika", "Malgova", "Mallika", "Neelam"]
ripeness = ["ripe", "Rotten", "unripe"]

CONFIDENCE_THRESHOLD = 0.6

def preprocess_image(image):
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_mango(image):
    processed_image = preprocess_image(image)
    
    category_prediction = category_model.predict(processed_image)
    ripeness_prediction = ripeness_model.predict(processed_image)
    
    category_index = np.argmax(category_prediction[0])
    ripeness_index = np.argmax(ripeness_prediction[0])
    
    if ripeness_prediction[0][ripeness_index] < CONFIDENCE_THRESHOLD:
        ripeness_index = -1
    
    detected_category = categories[category_index] if 0 <= category_index < len(categories) else "Unknown"
    detected_ripeness = ripeness[ripeness_index] if 0 <= ripeness_index < len(ripeness) else "Uncertain"
    
    return detected_category, detected_ripeness
def check_if_mango_exists(image):
    """
    Balanced mango detection that prevents false positives while maintaining
    flexibility for different mango varieties.
    """
    try:
        
        img_array = np.array(image)
        
        
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        
        lower_yellow = np.array([15, 70, 70])
        upper_yellow = np.array([35, 255, 255])
        
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 70, 70])
        upper_red2 = np.array([180, 255, 255])
        
        
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        
        
        combined_mask = yellow_mask + green_mask + red_mask1 + red_mask2
        
        
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
       
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if not contours:
            return False
            
        
        img_height, img_width = img_array.shape[:2]
        img_area = img_height * img_width
            
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        
        for contour in contours[:3]: 
            area = cv2.contourArea(contour)
            
            
            if area < 1000 or area > (img_area * 0.7):
                continue
                
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
           
            if 0.6 <= aspect_ratio <= 1.8:
                
                if len(contour) >= 5: 
                    ellipse = cv2.fitEllipse(contour)
                    ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
                    
                   
                    area_ratio = area / ellipse_area if ellipse_area > 0 else 0
                    if 0.7 <= area_ratio <= 1.3:
                        
                        contour_mask = np.zeros_like(cleaned_mask)
                        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                        
                        
                        contour_pixels = np.sum(contour_mask > 0)
                        mango_colored_pixels = np.sum((contour_mask > 0) & (combined_mask > 0))
                        color_percentage = (mango_colored_pixels / contour_pixels) * 100 if contour_pixels > 0 else 0
                        
                        
                        if color_percentage > 50:
                            return True
        
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        
        ksize = 15
        sigma = 4.0
        theta = 0
        lambd = 10.0
        gamma = 0.5
        
        
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        
        
        filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
        
        
        if len(contours) > 0 and cv2.contourArea(contours[0]) > 5000:
            
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contours[0]], 0, 255, -1)
            
            
            masked_texture = cv2.bitwise_and(filtered_img, filtered_img, mask=mask)
            
            
            if np.sum(mask > 0) > 0:
                texture_std = np.std(masked_texture[mask > 0])
                
                
                if 5 < texture_std < 40:
                    
                    color_percentage = (np.sum((mask > 0) & (combined_mask > 0)) / np.sum(mask > 0)) * 100
                    if color_percentage > 40:
                        return True
        
        
        return False
        
    except Exception as e:
        print(f"Error in mango detection: {e}")
        
        return False
def main():
    st.markdown('<h1 class="title">🥭 Mango Category & Ripeness Detector</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    option = st.radio("Choose an option:", ["Upload Image", "Use Webcam"])
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image of a mango", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            
            is_mango = check_if_mango_exists(image)
            
            if is_mango:
                category, ripeness_status = predict_mango(image)
                st.success(f"Detected Mango: {category}\nRipeness: {ripeness_status}")
            else:
                st.error("No mango detected in the image. Please upload an image containing a mango.")
            
    elif option == "Use Webcam":
        if st.button("Detect the Mango"):
            st.write("Opening webcam... Please press *Q* to capture your image.")
            cap = cv2.VideoCapture(0)
            detected_category = None
            detected_ripeness = None
            mango_detected = False
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video.")
                    break
                    
                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    
                    mango_detected = check_if_mango_exists(image)
                    
                    if mango_detected:
                        detected_category, detected_ripeness = predict_mango(image)
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            
            
            if mango_detected:
                if detected_category and detected_ripeness:
                    st.session_state.detected_category = detected_category
                    st.session_state.detected_ripeness = detected_ripeness
                    st.markdown(f"""
    <div class="result-container" style="background-color: black; padding: 10px; border-radius: 8px;">
        <h3 style="color: white;">Results:</h3>
        <p style="color: white;">Detected Category: {detected_category}</p>
        <p style="color: white;">Ripeness: {detected_ripeness}</p>
    </div>
""", unsafe_allow_html=True)

            else:
                st.error("No mango detected in the image. Please capture an image containing a mango.")   
            


if __name__ == "__main__":
    main()