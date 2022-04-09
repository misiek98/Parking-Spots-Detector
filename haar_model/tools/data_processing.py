import os

import cv2

from .other_functions import load_config

config = load_config(
    r'C:\Users\Misiek\Desktop\kekkek\Parking-Spots-Detector\haar_model\tools\haar.json')


class Point:
    """ 
    Class Point contains X and Y coordinates.

    x: x coordinate
    y: y coordinate
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class data_processing_methods(Point):
    """
    The methods below allow you to prepare data for a Haar-Cascade algorithm. 
    """

    def __init__(self, x, y):
        super().__init__(x, y)

    @staticmethod
    def prepare_vector_data(listOfPoints: list):
        """
        This method prepares data from listOfPoints to generate a .vec file that will be used to train a Haar Cascade model.

        listOfPoints: list that contains Point objects


        The list should consists of an even number of points (if not, there is an error). The iteration starts from the second element of the list. 
        To create an area, you need 2 points - an initial point with the smallest x and y coordinates (top left corner), and the width and height of that area. The width and height of the area are calculated and stored in a list at the iterated position. Finally, we go 2 steps further in the loop and repeat the operation.
        """

        for iteration, _ in enumerate(listOfPoints):
            if (iteration % 2 == 1):
                width = abs(listOfPoints[iteration].x
                            - listOfPoints[iteration - 1].x)
                height = abs(listOfPoints[iteration].y
                             - listOfPoints[iteration - 1].y)

                if (listOfPoints[iteration - 1].x > listOfPoints[iteration].x):
                    listOfPoints[iteration - 1].x = listOfPoints[iteration].x
                if (listOfPoints[iteration - 1].y > listOfPoints[iteration].y):
                    listOfPoints[iteration - 1].y = listOfPoints[iteration].y

                listOfPoints[iteration] = Point(width, height)

    @staticmethod
    def save_marked_area(listOfPoints: list, img: str):
        """
        This method cuts out the marked part(s) of the image and save it to neg folder.

        listOfPoints: list that contains Point objects
        img: copy of the image
        """

        for iteration, _ in enumerate(listOfPoints):
            if (iteration % 2 == 1):
                topLeftX = min(listOfPoints[iteration - 1].x,
                               listOfPoints[iteration].x)
                topLeftY = min(listOfPoints[iteration - 1].y,
                               listOfPoints[iteration].y)
                bottomRightX = max(listOfPoints[iteration - 1].x,
                                   listOfPoints[iteration].x)
                bottomRightY = max(listOfPoints[iteration - 1].y,
                                   listOfPoints[iteration].y)

                negativeImagesDir = config['negDir']

                cv2.imwrite(
                    filename=os.path.join(
                        negativeImagesDir,
                        f'file{len(os.listdir(negativeImagesDir))}.jpg'),
                    img=img[topLeftY:bottomRightY,
                            topLeftX:bottomRightX])

    @staticmethod
    def draw_rectangles(listOfPoints: list, img: str):
        """
        This method draws rectangles around marked objects.

        listOfPoints: list that contains Point objects
        img: copy of the image that you want to draw rectangles
        """

        for iteration, _ in enumerate(listOfPoints):
            if (iteration % 2 == 1):
                firstPoint = (
                    listOfPoints[iteration - 1].x,
                    listOfPoints[iteration - 1].y
                )
                secondPoint = (
                    listOfPoints[iteration].x,
                    listOfPoints[iteration].y
                )

                cv2.putText(
                    img=img,
                    text=str(int((iteration - 1)/2)),
                    org=(
                        (firstPoint[0] - 15),
                        (firstPoint[1]) - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(0, 100, 255),
                    thickness=2
                )

                image = cv2.rectangle(
                    img=img,
                    pt1=firstPoint,
                    pt2=secondPoint,
                    color=(0, 50, 255),
                    thickness=1
                )

        return image

    @staticmethod
    def remove_point(img: str, listOfPoints: list):
        """

        """

        listOfPoints.pop()

        for point in listOfPoints:
            cv2.circle(
                img=img,
                center=(point.x, point.y),
                radius=2,
                color=(255, 255, 255),
                thickness=-1)

        return img, listOfPoints


def get_position(event, x, y, flags, param):
    """
    Function get_position does a few things. When you double click left mouse button function will draw a small circle at this point and add their coordinates (saved in class Point) to the listOfPoints list.

    event: induced action
    x: x coordinate
    y: y coordinate
    flags and params were required to this function but weren't used.
    """
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y)
        cv2.circle(
            img=param[0],
            center=(x, y),
            radius=2,
            color=(255, 255, 255),
            thickness=-1)

        param[1].append(Point(x=x, y=y))
