# import streamlit as st
# from yolo_predictions import YOLO_Pred
# from PIL import Image
# import numpy as np

# st.set_page_config(page_title="YOLO Object Detection",
#                    layout='wide',
#                    page_icon='./images/nail.png')

# st.header('Welcome to Nail Health Portal')
# # st.write('Please Upload Image to get detections')

# with st.spinner('Please wait while your model is loading'):
#     yolo = YOLO_Pred(onnx_model=r'C:/Users/ripud/Desktop/OneDrive - UTS/UTS/Sem 3/DEEP_LEARNING/GUI/Ripu/Nail_Health/models/nail_yolo/best.onnx',
#                     data_yaml=r'C:/Users/ripud/Desktop/OneDrive - UTS/UTS/Sem 3/DEEP_LEARNING/GUI/Ripu/Nail_Health/models/nail_yolo/data.yaml')
                    



# col1, col2 = st.columns(2)
# with col1:
#     # Upload Image
#     image_file = st.file_uploader(label='Upload Image')
#     def upload_image():
#         if image_file is not None:
#             size_mb = image_file.size/(1024**2)
#             file_details = {"filename":image_file.name,
#                             "filetype":image_file.type,
#                             "filesize": "{:,.2f} MB".format(size_mb)}
#             #st.json(file_details)
#             # validate file
#             if file_details['filetype'] in ('image/png','image/jpeg'):
#                 st.success('VALID IMAGE file type (png or jpeg)')
#                 return {"file":image_file,
#                         "details":file_details}
            
#             else:
#                 st.error('INVALID Image file type')
#                 st.error('Upload only png,jpg, jpeg')
#                 return None

# with col2:
#     st.markdown("##### Take Picture #####")
#     buttonVal = st.button('Do you want to take picture from Camera?')
#     def camera_input():
#         if buttonVal:
#             picture = st.camera_input("Take a picture")
#             if picture:
#                 st.image(picture)
                
        
# def main():
#     object = upload_image()
#     object1 = camera_input()
#     if object:
#         prediction = False
#         image_obj = Image.open(object['file'])       
        
#         col1 , col2 = st.columns(2)
        
#         with col1:
#             st.info('Preview of Image')
#             st.image(image_obj)
            
#         with col2:
#             st.subheader('Check below for file details')
#             st.json(object['details'])
#             button = st.button('Get Detection from YOLO')
#             if button:
#                 with st.spinner("""
#                 Geting Objects from image. please wait
#                                 """):
#                     # below command will convert
#                     # obj to array
#                     image_array = np.array(image_obj)
#                     pred_img = yolo.predictions(image_array)
#                     pred_img_obj = Image.fromarray(pred_img)
#                     prediction = True
                
#         if prediction:
#             st.subheader("Predicted Image")
#             st.caption("Object detection from YOLO V5 model")
#             st.image(pred_img_obj)
    
#     # elif object1:
#     #     st.image(object1['Picture'])
#         # prediction = False
#         # image_obj = Image.open(object['file'])       
        
#         # col1 , col2 = st.columns(2)
        
#         # with col1:
#         #     st.info('Preview of Image')
#         #     st.image(image_obj)
            
#         # with col2:
#         #     st.subheader('Check below for file details')
#         #     st.json(object['details'])
#         #     button = st.button('Get Detection from YOLO')
#         #     if button:
#         #         with st.spinner("""
#         #         Geting Objets from image. please wait
#         #                         """):
#         #             # below command will convert
#         #             # obj to array
#         #             image_array = np.array(image_obj)
#         #             pred_img = yolo.predictions(image_array)
#         #             pred_img_obj = Image.fromarray(pred_img)
#         #             prediction = True
                
#         # if prediction:
#         #     st.subheader("Predicted Image")
#         #     st.caption("Object detection from YOLO V5 model")
#         #     st.image(pred_img_obj)
    
    
    
# if __name__ == "__main__":
#     main()






# import streamlit as st
# from yolo_predictions import YOLO_Pred
# from PIL import Image
# import numpy as np

# st.set_page_config(page_title="YOLO Object Detection",
#                    layout='wide',
#                    page_icon='./images/nail.png')

# st.header('Get Object Detection for any Image')
# st.write('*Please Upload Image to get detections*')

# with st.spinner('Please wait while your model is loading'):
#     yolo = YOLO_Pred(onnx_model=r'./models/nail_yolo/yolov5_200.onnx',
#                     data_yaml=r'./models/nail_yolo/data_final.yml')
#     #st.balloons()

# def upload_image():
#     # Upload Image
#     image_file = st.file_uploader(label='Upload Image')
#     if image_file is not None:
#         size_mb = image_file.size/(1024**2)
#         file_details = {"filename":image_file.name,
#                         "filetype":image_file.type,
#                         "filesize": "{:,.2f} MB".format(size_mb)}
#         #st.json(file_details)
#         # validate file
#         if file_details['filetype'] in ('image/png','image/jpeg'):
#             st.success('VALID IMAGE file type (png or jpeg')
#             return {"file":image_file,
#                     "details":file_details}
        
#         else:
#             st.error('INVALID Image file type')
#             st.error('Upload only png,jpg, jpeg')
#             return None
        
# def main():
#     object = upload_image()
    
#     if object:
#         prediction = False
#         image_obj = Image.open(object['file'])       
        
#         col1 , col2 = st.columns(2)
        
#         with col1:
#             st.info('Preview of Image')
#             st.image(image_obj)
            
#         with col2:
#             st.subheader('Check below for file details')
#             st.json(object['details'])
#             button = st.button('Get Detection from YOLO')
            
#             if button:
#                 with st.spinner("""
#                 Geting Objets from image. please wait
#                                 """):
#                     # below command will convert
#                     # obj to array
#                     image_array = np.array(image_obj)
#                     pred_img = yolo.predictions(image_array)
#                     pred_img_obj = Image.fromarray(pred_img)
#                     prediction = True
                
#         if prediction:
#             st.subheader("Predicted Image")
#             st.caption("Object detection from YOLO V5 model")
#             st.image(pred_img_obj)
    
    
    
# if __name__ == "__main__":
#     main()

# import streamlit as st
# from PIL import Image
# import numpy as np
# import torch
# from io import *
# import glob
# from datetime import datetime
# import os
# import wget

# st.set_page_config(page_title="YOLO Object Detection",
#                    layout='wide',
#                    page_icon='./images/nail.png')

# st.header('Get Object Detection for Nail Image')
# st.write('*Please Upload Image to get detections*')

# st.header('Get Object Detection for Nail Image')


# CFG_MODEL_PATH = "models/yolo_nail200.pt"
# deviceoption = "CPU"

# @st.cache_resource
# def loadmodel():
#     with st.spinner('Please wait while your model is loading'):
#         model = torch.hub.load('ultralytics/yolov5', 'custom', path=CFG_MODEL_PATH, force_reload=True, device=deviceoption)
#     return model
    
    

# def upload_image():
#     # Upload Image
#     image_file = st.file_uploader(label='Upload Image')
#     if image_file is not None:
#         size_mb = image_file.size/(1024**2)
#         file_details = {"filename":image_file.name,
#                         "filetype":image_file.type,
#                         "filesize": "{:,.2f} MB".format(size_mb)}
#         #st.json(file_details)
#         # validate file
#         if file_details['filetype'] in ('image/png','image/jpeg'):
#             st.success('VALID IMAGE file type (png or jpeg')
#             return {"file":image_file,
#                     "details":file_details}
        
#         else:
#             st.error('INVALID Image file type')
#             st.error('Upload only png,jpg, jpeg')
#             return None
        
# def main():
#     model = loadmodel()
#     object = upload_image()
    
    
#     if object:
#         prediction = False
#         image_obj = Image.open(object['file'])
#         img_details = object['details']       
#         # print("img_details :",img_details)
#         col1 , col2 = st.columns(2)
        
#         with col1:
#             st.info('Preview of Image')
#             st.image(image_obj,width=300)
        
#         ts = datetime.timestamp(datetime.now())
#         imgpath = os.path.join('data/uploads', str(ts)+img_details['filename'])  
#         outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
        
#         with open(imgpath, mode="wb") as f:
#             image_obj.save(f)
#             st.success("File saved successfully")

#         with col2:
#             st.subheader('Check below for file details')
#             st.json(object['details'])
#             button = st.button('Get Detection from YOLO')
            
#             if button:
#                 with st.spinner(""" Getting Objects from image. please wait """):
#                     pred = model(imgpath)
#                     # pred.dtype()
#                     # print(type(pred))
#                     # print("Pred-------------", pred)
#                     # names = pred.split(" ")
#                     # print("names",names)
#                     pred.render()
#                         # save output to file
#                     for im in pred.ims:
#                         im_base64 = Image.fromarray(im)
#                         im_base64.save(outputpath)
#                     prediction = True
                
#         if prediction:
#             # Predictions
#             img_ = Image.open(outputpath)
#             st.image(img_, caption='Model Prediction(s)', width=300)
            
    
    
# if __name__ == "__main__":
#     main()

import streamlit as st
from PIL import Image
import torch
from datetime import datetime
import os
import requests

# ========== CONFIG ==========
GOOGLE_DRIVE_FILE_ID = '14h3USnlg9sbDtowqFFo6m_XG4y0nRY4z'  # <-- Replace with your actual file ID
MODEL_PATH = 'yolo_nail200.pt'
DEVICE_OPTION = 'cpu'

# ========== PAGE SETUP ==========
st.set_page_config(page_title="YOLO Nail Detection", layout='wide', page_icon='./images/nail.png')
st.header('Get Object Detection for Nail Image')

st.text("Please refer the sample images below for taking a photo (only raw unmanicured nails):")
col1, col2, col3 = st.columns(3)
with col1:
    st.image('./images/VALIDPITCURE/pic3.jpg', width=200)
with col2:
    st.image('./images/VALIDPITCURE/pic1.jpg', width=200)
with col3:
    st.image('./images/VALIDPITCURE/pic2.jpg', width=200)

st.warning("This detection is not medical advice. For health concerns, consult a doctor.")

# ========== CREATE FOLDERS ==========
os.makedirs('data/uploads', exist_ok=True)
os.makedirs('data/outputs', exist_ok=True)

# ========== DOWNLOAD MODEL ==========
def download_from_gdrive(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# ========== LOAD YOLO MODEL ==========
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading YOLO model...'):
            download_from_gdrive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
            st.success('Model downloaded successfully.')

    with st.spinner('Loading model...'):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True, device=DEVICE_OPTION)
    return model

# ========== IMAGE UPLOAD ==========
def upload_image():
    image_file = st.file_uploader("Upload an image to get detection", type=['png', 'jpg', 'jpeg'])
    if image_file:
        return {
            'file': image_file,
            'name': image_file.name,
            'type': image_file.type,
            'size': f"{image_file.size / (1024**2):.2f} MB"
        }
    return None

# ========== MAIN ==========
def main():
    model = load_model()
    image_data = upload_image()

    if image_data:
        image = Image.open(image_data['file']).convert('RGB')
        timestamp = datetime.now().timestamp()
        filename = f"{int(timestamp)}_{image_data['name']}"

        upload_path = f"data/uploads/{filename}"
        output_path = f"data/outputs/{filename}"

        # Save original image
        image.save(upload_path)
        st.success("Image uploaded successfully.")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", width=300)
        with col2:
            st.json({
                "Filename": image_data['name'],
                "Type": image_data['type'],
                "Size": image_data['size']
            })

            if st.button("Run YOLO Detection"):
                with st.spinner("Detecting..."):
                    pred = model(upload_path)
                    pred.render()

                    for im in pred.ims:
                        im_output = Image.fromarray(im)
                        im_output.save(output_path)

                st.success("Detection complete!")
                st.image(Image.open(output_path), caption="Prediction", width=500)

if __name__ == "__main__":
    main()
