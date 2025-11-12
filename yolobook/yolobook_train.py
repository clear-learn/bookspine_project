from ultralytics import YOLO

model = YOLO("paste .pt file path")  # load trained model
model.resume = True

results = model.train(data="yaml path", epochs=200, imgsz=864, device =[1,0], plots=True, overlap_mask= False, single_cls= False, batch = 20, save_period = 1, degrees= 90, shear = 15, exist_ok = False)
