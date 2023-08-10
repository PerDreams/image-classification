from ultralyticsplus import YOLO, render_result

# load model
model = YOLO('ultralyticsplus/yolov8s')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
image = 'lmao2.jpeg'  # or 'https://ultralytics.com/images/zidane.jpg'

# perform inference
results = model.predict(image)

# observe results
print(results[0].boxes)
#render = render_result(model=model, image=image, result=results[0])
#render.show()

