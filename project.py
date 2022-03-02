from haar_model.tools.data_processing import Point, data_processing_methods
import cv2
import numpy as np

with open(r'C:\Users\Misiek\Desktop\Python\MGR\yolov3_files\\coco.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet(r'C:\Users\Misiek\Desktop\Python\MGR\yolov3_files\\yolov3.cfg',
                                     r'C:\Users\Misiek\Desktop\Python\MGR\yolov3_files\\yolov3.weights')


probability_minimum = 0.5
threshold = 0.3
colours = np.random.randint(
    0, 255, size=(len(labels), 3), dtype='uint8')


listOfPoints = []


def get_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        listOfPoints.append(Point(x, y))
        cv2.circle(img=img, center=(x, y), radius=2,
                   color=(255, 255, 255), thickness=-1)

        print(f'x: {x}, y: {y}')


video = cv2.VideoCapture(
    r'C:\Users\Misiek\Desktop\Python\sample_video.mp4')
frameNumber = 1

while True:
    _, frame = video.read()
    img = frame.copy()

    if frameNumber == 2:
        '''
        predykcja liczby pasów
        ustawienie poprawnej liczby linii
        wyświetlenie liczby miejsc parkingowych
        '''
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', get_position)

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lineDetector = cv2.CascadeClassifier(
            r'C:\Users\Misiek\Desktop\Python\MGR\haar_model\results\cascade.xml')

        lineRegions = lineDetector.detectMultiScale(grayImage, scaleFactor=1.13, minNeighbors=3, minSize=(
            105, 27),  maxSize=(270, 80), flags=cv2.CASCADE_SCALE_IMAGE)

        for i, (x, y, w, h) in enumerate(lineRegions):
            cv2.putText(img, str(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 255, 0), thickness=2)
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (0, 100, 255), 1)

        while True:
            cv2.imshow('Frame', img)

            if cv2.waitKey(1) == ord('w'):
                '''
                okienko z pytaniem o index do usunięcia
                rysowanie na nowo
                '''
                img = frame.copy()
                userInput = int(input('Choose a rectangle to drop: '))
                lineRegions = np.delete(lineRegions, userInput, axis=0)

                for i, (x, y, w, h) in enumerate(lineRegions):
                    cv2.putText(img, str(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=.5, color=(0, 255, 0), thickness=2)
                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  (0, 100, 255), 1)

                print(
                    f'Rectangle number {userInput} has been successfully removed!')

            # remove marked point
            if cv2.waitKey(1) == ord('r'):
                img = frame.copy()
                listOfPoints.pop()
                for point in listOfPoints:
                    cv2.circle(img=img, center=(
                        point.x, point.y), radius=2, color=(255, 255, 255), thickness=-1)

            # draw new rectangles
            if cv2.waitKey(1) == ord('d'):
                img = frame.copy()
                data_processing_methods.prepare_vector_data(listOfPoints)

                for i, _ in enumerate(listOfPoints):
                    if i % 2 == 1:
                        lineRegions = np.append(lineRegions,
                                                [[listOfPoints[i - 1].x,  listOfPoints[i - 1].y, listOfPoints[i].x, listOfPoints[i].y]], axis=0)

                for i, (x, y, w, h) in enumerate(lineRegions):
                    cv2.putText(img, str(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=.5, color=(0, 255, 0), thickness=2)
                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  (0, 100, 255), 1)

            # next
            if cv2.waitKey(1) == ord('n'):
                cv2.destroyAllWindows()
                frameNumber += 1
                break

    if frameNumber % 30 == 0:
        h, w = None, None
        layers_names_all = network.getLayerNames()
        layers_names_output = \
            [layers_names_all[i - 1]
                for i in network.getUnconnectedOutLayers()]

        if w is None or h is None:
            h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)

        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)

        bounding_boxes = []
        confidences = []
        classIDs = []

        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:

                    box_current = detected_objects[0:4] * \
                        np.array([w, h, w, h])

                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append(
                        [x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    classIDs.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                   probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                colour_box_current = colours[classIDs[i]].tolist()

                cv2.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                text_box_current = '{}: {:.4f}'.format(labels[int(classIDs[i])],
                                                       confidences[i])

                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

    frameNumber += 1
