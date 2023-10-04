from ultralytics import YOLO


# Load a model
#model = YOLO("yolov8l.yaml")  # build a new model from scratch
model = YOLO("models/yolov8l.pt")  # load a pretrained model

model.train(data="../datasets/wood-fs/wood.yaml", epochs=100, batch=12, imgsz=1088, device=0)

#results = model.predict(source = "/gpfs/scratch/rayen/YOLOv8/image.png", save = True)  # predict on an image
#results = model.predict(source = "/gpfs/scratch/rayen/YOLOv8/image2.png", save = True)
#results = model.predict(source = "/gpfs/scratch/rayen/YOLOv8/image3.png", save = True)

#path = model.export(format="onnx")  # export the model to ONNX format
