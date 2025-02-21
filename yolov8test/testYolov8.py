from ultralytics import YOLO
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

# Load a model
model = YOLO("yolov8n.pt")  # load an official model

if __name__ == '__main__':
    # # print(r"\\tlprr-final.v3i.yolov8\\data.yaml")
    # results = model.train(data=r"F:\\PythonProject\\ParkingMonitor\\tlprr-final.v3i.yolov8\\data.yaml", epochs=3)
    # # Validate the model
    # results = model.val(data=r"F:\\PythonProject\\ParkingMonitor\\tlprr-final.v3i.yolov8\\data.yaml")  
    # # no arguments needed, dataset and settings remembered
    
    
    
    # Load a model
    model = YOLO("yolov8n.pt")  # load an official model
    model = YOLO("runs/detect/train13/weights/best.pt")  # load a custom model
    # Predict with the model
    results = model.predict('car-plate1.jpg',save=True)  # predict on an image