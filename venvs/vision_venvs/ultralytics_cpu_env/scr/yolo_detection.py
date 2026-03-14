from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
results = model("./data/2.jpg")

# Get annotated image
annotated_img = results[0].plot()

# Show
cv2.imshow("YOLO Result", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#for r in results:
#    print(r.boxes.cls)