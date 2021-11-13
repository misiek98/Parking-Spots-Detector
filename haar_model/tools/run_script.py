import os
import cv2
import shutil
from data_processing import Point, data_processing_methods


listOfPoints = []


def get_position(event, x, y, flags, param):
    '''
    Function get_position does a few things. When you double click left mouse button function will draw a small circle at this point and add their coordinates (saved in class Point) to the listOfPoints list.

    event: induced action
    x: x coordinate
    y: y coordinate
    flags and params were required to this function but weren't used.
    '''
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img=img, center=(x, y), radius=2,
                   color=(255, 255, 255), thickness=-1)
        listOfPoints.append(Point(x=x, y=y))


imageSet = os.path.join(
    os.getcwd(), 'haar_model', 'data')


for image in os.listdir(imageSet):
    imagePath = os.path.join(imageSet, image)
    img = cv2.imread(imagePath)

    cv2.namedWindow('Parking space')
    cv2.setMouseCallback('Parking space', get_position)

    while True:
        cv2.imshow('Parking space', img.copy())

        # by pressing 'n' program process data from listOfPoints, save it to parking_lines.info file and copy image from data to pos folder
        if cv2.waitKey(1) == ord('n'):
            if len(listOfPoints) % 2 == 0:
                print('Correct number of points, creating file...')

                data_processing_methods.prepare_vector_data(
                    listOfPoints=listOfPoints)

                pathToInfoFile = os.path.join(
                    os.getcwd(), 'haar_model', 'parking_lines.info')

                posImagePath = os.path.join(
                    os.getcwd(), 'haar_model', 'pos', image)

                # save processed data to parking_lines.info file
                with open(pathToInfoFile, 'a') as file:
                    file.write(f'{posImagePath} {int(len(listOfPoints) / 2)} ')

                    for coordinate in listOfPoints:
                        file.write(f'{coordinate.x} {coordinate.y} ')
                    file.write('\n')

                print(f'File was saved: {pathToInfoFile}')
                listOfPoints.clear()

                # copies image from data folder to pos folder
                shutil.copyfile(src=imagePath, dst=os.path.join(
                    os.getcwd(), 'haar_model', 'pos', image))

                break

            else:
                print('Incorrect number of points, points weren\'t saved')
                listOfPoints.clear()
                break

        # by pressing 'c' program cuts out the marked part(s) of the image and saves it to the neg folder.
        if cv2.waitKey(1) == ord('c'):
            img = cv2.imread(imagePath)
            data_processing_methods.save_marked_area(
                listOfPoints=listOfPoints, img=img)
            listOfPoints.clear()
            break

        # by pressing 'd' program displays image with drawn rectangles around marked objects
        if cv2.waitKey(1) == ord('d'):
            imageWithRectangles = cv2.imread(imagePath)
            cv2.imshow('Marked objects',
                       data_processing_methods.draw_rectangles(listOfPoints=listOfPoints, img=imageWithRectangles))

        # by pressing 'r' program removes last point from listOfPoints and from image
        if cv2.waitKey(1) == ord('r'):
            img = cv2.imread(imagePath)
            listOfPoints.pop()
            for point in listOfPoints:
                cv2.circle(img=img, center=(
                    point.x, point.y), radius=2, color=(255, 255, 255), thickness=-1)
