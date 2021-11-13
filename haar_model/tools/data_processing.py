import os
import cv2


class Point:
    '''
    Class Point contains X and Y coordinates.

    x: x coordinate
    y: y coordinate
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y


class data_processing_methods(Point):
    '''
    The methods below allow you to prepare data for a Haar-Cascade algorithm. 
    '''

    def __init__(self, x, y):
        super().__init__(x, y)

    @staticmethod
    def prepare_vector_data(listOfPoints: list):
        '''
        This method prepares data from listOfPoints to generate a .vec file that will be used to train a Haar Cascade model.

        listOfPoints: list that contains Point objects
        '''

        for iteration in range(len(listOfPoints)):
            if iteration % 2 == 1:
                width = abs(listOfPoints[iteration].x -
                            listOfPoints[iteration - 1].x)

                height = abs(listOfPoints[iteration].y -
                             listOfPoints[iteration - 1].y)

                if listOfPoints[iteration - 1].x > listOfPoints[iteration].x:
                    listOfPoints[iteration - 1].x = listOfPoints[iteration].x

                if listOfPoints[iteration - 1].y > listOfPoints[iteration].y:
                    listOfPoints[iteration - 1].y = listOfPoints[iteration].y

                listOfPoints[iteration] = Point(width, height)

    @staticmethod
    def save_marked_area(listOfPoints, img):
        '''
        This method cuts out the marked part(s) of the image and save it to neg folder.

        listOfPoints: list that contains Point objects
        img: copy of the image
        '''

        for iteration in range(len(listOfPoints)):
            if iteration % 2 == 1:
                topLeftX = min(listOfPoints[iteration - 1].x,
                               listOfPoints[iteration].x)
                topLeftY = min(listOfPoints[iteration - 1].y,
                               listOfPoints[iteration].y)

                bottomRightX = max(listOfPoints[iteration - 1].x,
                                   listOfPoints[iteration].x)
                bottomRightY = max(listOfPoints[iteration - 1].y,
                                   listOfPoints[iteration].y)

                negativeImagesDir = os.path.join(
                    os.getcwd(), 'haar_model', 'neg')

                cv2.imwrite(os.path.join(
                    negativeImagesDir, f'file{len(os.listdir(negativeImagesDir))}.jpg'), img=img[topLeftY:bottomRightY, topLeftX:bottomRightX])

    @staticmethod
    def draw_rectangles(listOfPoints, img):
        '''
        This method draws rectangles around marked objects.

        listOfPoints: list that contains Point objects
        img: copy of the image that you want to draw rectangles
        '''

        for iteration in range(len(listOfPoints)):
            if iteration % 2 == 1:
                firstPoint = (
                    listOfPoints[iteration - 1].x, listOfPoints[iteration - 1].y)
                secondPoint = (
                    listOfPoints[iteration].x, listOfPoints[iteration].y)

                image = cv2.rectangle(img=img, pt1=firstPoint,
                                      pt2=secondPoint, color=(0, 0, 255), thickness=2)

        return image
