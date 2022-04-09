import os

import cv2
import shutil

from data_processing import data_processing_methods, get_position
from other_functions import load_config

config = load_config(
    r'C:\Users\Misiek\Desktop\kekkek\Parking-Spots-Detector\haar_model\tools\haar.json')

listOfPoints = []
imageSet = config['data']
pathToInfoFile = config['parkingLinesInfoFile']
posImagePath = config['posDir']

for image in os.listdir(imageSet):
    imagePath = os.path.join(
        imageSet,
        image
    )
    img = cv2.imread(imagePath)

    while True:
        cv2.namedWindow('Parking space')
        cv2.setMouseCallback(
            window_name='Parking space',
            on_mouse=get_position,
            param=[img, listOfPoints]
        )

        cv2.imshow('Parking space', img.copy())

        # by pressing 'n' program process data from listOfPoints,
        # save it to parking_lines.info file and copy image from data dir to pos folder
        if cv2.waitKey(1) == ord('n'):
            try:
                assert len(listOfPoints) % 2 == 0

                data_processing_methods.prepare_vector_data(
                    listOfPoints=listOfPoints
                )

                with open(pathToInfoFile, 'a') as file:
                    file.write(f'{posImagePath} {int(len(listOfPoints) / 2)} ')
                    for coordinate in listOfPoints:
                        file.write(f'{coordinate.x} {coordinate.y} ')
                    file.write('\n')

                listOfPoints.clear()
                print(f'File was saved: {pathToInfoFile}')

                shutil.copyfile(
                    src=imagePath,
                    dst=os.path.join(
                        posImagePath,
                        image
                    )
                )
                break

            except AssertionError:
                print("Insufficient number of points, data wasn't saved")
                listOfPoints.clear()
                break

        # by pressing 'c' program cuts out the marked part(s) of the image and saves it to the neg folder.
        if cv2.waitKey(1) == ord('c'):
            try:
                assert len(listOfPoints) % 2 == 0

                img = cv2.imread(imagePath)
                data_processing_methods.save_marked_area(
                    listOfPoints=listOfPoints,
                    img=img
                )
                listOfPoints.clear()
                break

            except AssertionError:
                print("Insufficient number of points, images wasn't saved")
                listOfPoints.clear()
                break

        # by pressing 'd' program displays image with drawn rectangles around marked objects
        if cv2.waitKey(1) == ord('d'):
            try:
                assert len(listOfPoints) % 2 == 0

                imageWithRectangles = cv2.imread(imagePath)
                cv2.imshow(
                    'Marked objects',
                    data_processing_methods.draw_rectangles(
                        listOfPoints=listOfPoints,
                        img=imageWithRectangles
                    )
                )

            except AssertionError:
                print('Insufficient number of points, rectangles cannot be drawn.')

        # by pressing 'r' program removes last point from listOfPoints and from image
        if cv2.waitKey(1) == ord('r'):
            img = cv2.imread(imagePath)
            img, listOfPoints = data_processing_methods.remove_point(
                img=img,
                listOfPoints=listOfPoints
            )
