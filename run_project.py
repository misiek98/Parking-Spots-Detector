import cv2
import numpy as np

from haar_model.tools.data_processing import data_processing_methods, Point, get_position
from haar_model.tools.other_functions import load_config

config = load_config(
    r'C:\Users\Misiek\Desktop\Python\MGR\project_config.json'
)

with open(config['yoloNames']) as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet(
    config['yoloCfg'],
    config['yoloWeights']
)

probability_minimum = config['probabilityMinimum']
threshold = config['threshold']
video = cv2.VideoCapture(config['video'])

colours = np.random.randint(
    low=0,
    high=255,
    size=(len(labels), 3),
    dtype='uint8'
)

listOfPoints = []
listOfParkingAreas = []

frameNumber = 1
while True:
    ret, frame = video.read()
    if ret == False:
        cv2.waitKey(0)
        # break

    img = frame.copy()

    if (frameNumber == 2):
        grayImage = cv2.cvtColor(
            src=img,
            code=cv2.COLOR_BGR2GRAY
        )
        lineDetector = cv2.CascadeClassifier(
            config['cascadeXML']
        )
        lineRegions = lineDetector.detectMultiScale(
            image=grayImage,
            scaleFactor=1.13,
            minNeighbors=3,
            minSize=(105, 27),
            maxSize=(270, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Change from numpy array to my concept.
        for x, y, w, h in lineRegions:
            listOfPoints.append(Point(x, y))
            listOfPoints.append(Point(x+w, y+h))

        data_processing_methods.draw_rectangles(
            listOfPoints=listOfPoints,
            img=img
        )

        while True:
            cv2.namedWindow('Frame')
            cv2.setMouseCallback(
                window_name='Frame',
                on_mouse=get_position,
                param=[img, listOfPoints]
            )
            cv2.imshow('Frame', img)

            # Press "w" to remove detected objects.
            # As an input take an rectangle(s) id separated by spaces.
            if (cv2.waitKey(1) == ord('w')):
                img = frame.copy()

                userInput = input(
                    'Choose a rectangle(s) id to drop (separated by spaces): ')
                userInput = [int(number) for number in userInput.split()]
                userInput = sorted(userInput, reverse=True)

                for idx in userInput:
                    listOfPoints.pop(2*idx + 1)
                    listOfPoints.pop(2*idx)

                data_processing_methods.draw_rectangles(
                    listOfPoints=listOfPoints,
                    img=img
                )
                print(
                    f'Rectangle(s) number {userInput} have been successfully removed!')

            # Press "r" to remove last marked point.
            if (cv2.waitKey(1) == ord('r')):
                img = frame.copy()

                img, listOfPoints = data_processing_methods.remove_point(
                    img=img,
                    listOfPoints=listOfPoints
                )

                if len(listOfPoints) % 2 != 0:
                    data_processing_methods.draw_rectangles(
                        listOfPoints=listOfPoints[:-1],
                        img=img
                    )
                else:
                    data_processing_methods.draw_rectangles(
                        listOfPoints=listOfPoints,
                        img=img
                    )

            # Press "d" to draw manually new rectangle(s).
            if (cv2.waitKey(1) == ord('d')):
                img = frame.copy()

                data_processing_methods.draw_rectangles(
                    listOfPoints=listOfPoints,
                    img=img
                )

            # Press "n" to mark parking areas.
            # As an input take an extreme rectangle(s).
            # After that, the program starts measuring the occupancy of parking spaces
            if (cv2.waitKey(1) == ord('n')):
                if (isinstance(listOfParkingAreas, np.ndarray)):
                    listOfParkingAreas = []

                img = frame.copy()

                userInput = input(
                    'Choose a rectangle(s) to mark parking area(s). Separate each ID with spaces: ')
                userInput = [int(number) for number in userInput.split()]

                tempListThatCOntainsPoints = []

                for idx in userInput:
                    tempListThatCOntainsPoints.append(listOfPoints[2 * idx])
                    tempListThatCOntainsPoints.append(listOfPoints[2*idx + 1])

                for i, _ in enumerate(tempListThatCOntainsPoints):
                    if i % 2 == 1:
                        listOfParkingAreas.append(
                            [max(tempListThatCOntainsPoints[i - 1].x, tempListThatCOntainsPoints[i].x),
                             min(tempListThatCOntainsPoints[i - 1].y, tempListThatCOntainsPoints[i].y)])
                        listOfParkingAreas.append(
                            [min(tempListThatCOntainsPoints[i - 1].x, tempListThatCOntainsPoints[i].x),
                             max(tempListThatCOntainsPoints[i - 1].y, tempListThatCOntainsPoints[i].y)])

                listOfParkingAreas = np.array(listOfParkingAreas)
                listOfParkingAreas = listOfParkingAreas.reshape(-1, 4, 2)

                for dim in listOfParkingAreas:
                    dim[[2, 3]] = dim[[3, 2]]

                cv2.destroyAllWindows()
                frameNumber += 1
                break

    if (frameNumber % 60 == 0):
        h, w = None, None
        layers_names_all = network.getLayerNames()
        layers_names_output = \
            [layers_names_all[i - 1]
                for i in network.getUnconnectedOutLayers()]

        if w is None or h is None:
            h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image=frame,
            scalefactor=1 / 255.,
            size=(416, 416),
            swapRB=True,
            crop=False
        )

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
                    x_min = int(x_center - (box_width//2))
                    y_min = int(y_center - (box_height//2))

                    bounding_boxes.append(
                        [x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    classIDs.append(class_current)

        results = cv2.dnn.NMSBoxes(
            bounding_boxes,
            confidences,
            probability_minimum,
            threshold
        )

        numberOfCars = 0
        if (len(results) > 0):
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                objectsCentralPoint = (
                    x_min + box_width//2,
                    y_min + box_height//2
                )

                for area in listOfParkingAreas:
                    isInParkingArea = cv2.pointPolygonTest(
                        contour=area,
                        pt=objectsCentralPoint,
                        measureDist=False
                    )
                    if (isInParkingArea == 1):
                        numberOfCars += 1
                        break

                # colour_box_current = colours[classIDs[i]].tolist()
                # cv2.rectangle(
                #     frame,
                #     (x_min, y_min),
                #     (x_min + box_width, y_min + box_height),
                #     colour_box_current,
                #     2
                # )
                # text_box_current = '{}: {:.4f}'.format(
                #     labels[int(classIDs[i])],
                #     confidences[i]
                # )
                # cv2.putText(
                #     frame,
                #     text_box_current,
                #     (x_min, y_min - 5),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #     colour_box_current,
                #     2
                # )

        frame[:120, -500:] = 0

        totalNumberOfParkingSpots = (len(listOfPoints)//2
                                     - len(listOfParkingAreas))
        occupiedParkingSpots = numberOfCars
        freeParkingSpots = totalNumberOfParkingSpots - occupiedParkingSpots

        cv2.putText(
            img=frame,
            text=f'Total number of parking spaces: {totalNumberOfParkingSpots}',
            org=(frame.shape[1]-480, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(100, 100, 255),
            thickness=2
        )
        cv2.putText(
            img=frame,
            text=f'Number of free parking spaces: {freeParkingSpots}',
            org=(frame.shape[1]-480, 65),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(100, 255, 0),
            thickness=2
        )
        cv2.putText(
            img=frame,
            text=f'Number of occupied parking spaces: {occupiedParkingSpots}',
            org=(frame.shape[1]-480, 105),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 255),
            thickness=2
        )

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

    frameNumber += 1
