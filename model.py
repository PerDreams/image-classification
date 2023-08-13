from ultralyticsplus import YOLO, render_result
import collections, os, shutil

# get class name with most common
def get_result_cls(results):
    cls = []

    for c in results[0].boxes.cls:
        cls.append(names[int(c)])
    
    counter = collections.Counter(cls)

    if counter.most_common(1) == []:
        return []

    return counter.most_common(1)[0][0]

# copy the image based on the class
def copy_image(cls, image):
    if os.path.isdir('../classified/' + cls) == False:
        os.mkdir('../classified/' + cls)
    shutil.copy('../Camera/'+image, '../classified/'+cls+'/'+image)

# load model
def load_model():
    model = YOLO('ultralyticsplus/yolov8s')

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    return model

# load the model
model = load_model()
names = model.names

counter = 0

# loop through images and get results
for img in os.listdir('../Camera/'):
    if img.endswith('.jpg'):
        image = '../Camera/' + img

        # perform inference
        results = model.predict(image)

        results_cls = get_result_cls(results)

        if results_cls != []:
            copy_image(results_cls, img)

# observe results
#render = render_result(model=model, image=image, result=results[0])
#render.show()