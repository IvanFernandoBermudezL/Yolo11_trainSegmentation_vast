from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8 model
model = YOLO(" ")

def main():
    # Train 
    model.train(
    data=" ", 
    epochs=120, 
    batch=8, 
    imgsz=640, 
    save_period=5)

if __name__ == '__main__':
    main()
