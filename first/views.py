import csv
import io
import pandas as pd
import numpy as np
from django.db import connection
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, get_object_or_404, get_list_or_404, redirect
from django.urls import reverse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from .models import *

from datetime import datetime, timedelta
from django.utils.dateformat import DateFormat

from multiprocessing import Process

#Create your views here.

def start(request):
    persons = Person.objects.filter(id=3800)
    # try:
    #     cursor = connection.cursor()
    #
    #     sql = "UPDATE `kioskdata`.`person` SET `Phone` = '0' WHERE (`ID` = '3800')"
    #     result = cursor.execute(sql)
    #     datas = cursor.fetchall()
    #
    #     connection.commit()
    #     connection.close()
    # except:
    #     connection.rollback()
    #     print("Failed")

    context = {'persons': persons}
    return render(request, 'first/start_page.html', context)

def menu(request):
    return render(request, 'first/kiosk_menu.html')

def capture(request):
    import cv2 as cv
    import numpy as np
    import os
    import pymysql

    def detect(img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    def removeFaceAra(img, cascade):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)

        height, width = img.shape[:2]

        for x1, y1, x2, y2 in rects:
            cv.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

        return img

    def make_mask_image(img_bgr):
        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

        # img_h,img_s,img_v = cv.split(img_hsv)

        low = (0, 30, 0)
        high = (15, 255, 255)

        img_mask = cv.inRange(img_hsv, low, high)
        return img_mask

    def distanceBetweenTwoPoints(start, end):
        x1, y1 = start
        x2, y2 = end

        return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))

    def calculateAngle(A, B):
        A_norm = np.linalg.norm(A)
        B_norm = np.linalg.norm(B)
        C = np.dot(A, B)

        angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
        return angle

    def findMaxArea(contours):
        max_contour = None
        max_area = -1

        for contour in contours:
            area = cv.contourArea(contour)

            x, y, w, h = cv.boundingRect(contour)

            if (w * h) * 0.4 > area:
                continue

            if w > h:
                continue

            if area > max_area:
                max_area = area
                max_contour = contour

        if max_area < 10000:
            max_area = -1

        return max_area, max_contour

    def getFingerPosition(max_contour, img_result, debug):
        points1 = []

        # STEP 6-1
        M = cv.moments(max_contour)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
        hull = cv.convexHull(max_contour)

        for point in hull:
            if cy > point[0][1]:
                points1.append(tuple(point[0]))

        if debug:
            cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
            for point in points1:
                cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

        # STEP 6-2
        hull = cv.convexHull(max_contour, returnPoints=False)
        defects = cv.convexityDefects(max_contour, hull)

        if defects is None:
            return -1, None

        points2 = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])

            angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

            if angle < 90:
                if start[1] < cy:
                    points2.append(start)

                if end[1] < cy:
                    points2.append(end)

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
            for point in points2:
                cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

        # STEP 6-3
        points = points1 + points2
        points = list(set(points))

        # STEP 6-4
        new_points = []
        for p0 in points:

            i = -1
            for index, c0 in enumerate(max_contour):
                c0 = tuple(c0[0])

                if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                    i = index
                    break

            if i >= 0:
                pre = i - 1
                if pre < 0:
                    pre = max_contour[len(max_contour) - 1][0]
                else:
                    pre = max_contour[i - 1][0]

                next = i + 1
                if next > len(max_contour) - 1:
                    next = max_contour[0][0]
                else:
                    next = max_contour[i + 1][0]

                if isinstance(pre, np.ndarray):
                    pre = tuple(pre.tolist())
                if isinstance(next, np.ndarray):
                    next = tuple(next.tolist())

                angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

                if angle < 90:
                    new_points.append(p0)

        return 1, new_points

    def process(img_bgr, debug):
        img_result = img_bgr.copy()

        # STEP 1
        img_bgr = removeFaceAra(img_bgr, cascade)

        # STEP 2
        img_binary = make_mask_image(img_bgr)

        # STEP 3
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
        cv.imshow("Binary", img_binary)

        # STEP 4
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if debug:
            for cnt in contours:
                cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

                # STEP 5
        max_area, max_contour = findMaxArea(contours)

        # if max_area == -1:
        # return img_result

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

            # STEP 6
        ret, points = getFingerPosition(max_contour, img_result, debug)

        finger_count = 0
        try:
            finger_count = len(points)
        except Exception as ex:
            print()
        # STEP 7
        # if ret > 0 and len(points) > 0:
        # for point in points:
        # cv.circle(img_result, point, 20, [255, 0, 255], 5)
        return finger_count

    current_file_path = os.path.dirname(os.path.realpath(__file__))
    cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))

    # cap = cv.VideoCapture('test.avi')

    cap = cv.VideoCapture(0)

    while True:

        ret, img_bgr = cap.read()

        if ret == False:
            break

        img_result = process(img_bgr, debug=False)

        key = cv.waitKey(1)
        if key == 27:
            break
        # cv.imshow("Result", img_result)

        if img_result == 1:
            return render(request, 'first/second_page.html')
        if img_result == 2:
            return render(request, 'first/second_page1.html')
        if img_result == 3:
            return render(request, 'first/second_page2.html')
        if img_result == 4:
            return render(request, 'first/second_page3.html')
        if img_result == 5:
            return render(request, 'first/third_page.html')

        # if img_result == 1:
        #     return render(request, 'first/second_page.html')
        # if img_result == 2:
        #     return render(request, 'first/second_page2.html')
        # if img_result == 3:
        #     return render(request, 'first/second_page2.html')
        # if img_result == 4:
        #     return render(request, 'first/second_page2.html')
        # if img_result == 5:
        #     return render(request, 'first/second_page2.html')

    cap.release()
    cv.destroyAllWindows()
    # return render(request, 'first/second_page.html')

def second_page(request):
    return render(request, 'first/second_page.html')

def second_page1(request):
    return render(request, 'first/second_page1.html')

def second_page2(request):
    return render(request, 'first/second_page2.html')

def second_page3(request):
    return render(request, 'first/second_page3.html')

def capture_burger(request):
    import cv2 as cv
    import numpy as np
    import os
    import pymysql

    def detect(img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    def removeFaceAra(img, cascade):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)

        height, width = img.shape[:2]

        for x1, y1, x2, y2 in rects:
            cv.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

        return img

    def make_mask_image(img_bgr):
        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

        # img_h,img_s,img_v = cv.split(img_hsv)

        low = (0, 30, 0)
        high = (15, 255, 255)

        img_mask = cv.inRange(img_hsv, low, high)
        return img_mask

    def distanceBetweenTwoPoints(start, end):
        x1, y1 = start
        x2, y2 = end

        return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))

    def calculateAngle(A, B):
        A_norm = np.linalg.norm(A)
        B_norm = np.linalg.norm(B)
        C = np.dot(A, B)

        angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
        return angle

    def findMaxArea(contours):
        max_contour = None
        max_area = -1

        for contour in contours:
            area = cv.contourArea(contour)

            x, y, w, h = cv.boundingRect(contour)

            if (w * h) * 0.4 > area:
                continue

            if w > h:
                continue

            if area > max_area:
                max_area = area
                max_contour = contour

        if max_area < 10000:
            max_area = -1

        return max_area, max_contour

    def getFingerPosition(max_contour, img_result, debug):
        points1 = []

        # STEP 6-1
        M = cv.moments(max_contour)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
        hull = cv.convexHull(max_contour)

        for point in hull:
            if cy > point[0][1]:
                points1.append(tuple(point[0]))

        if debug:
            cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
            for point in points1:
                cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

        # STEP 6-2
        hull = cv.convexHull(max_contour, returnPoints=False)
        defects = cv.convexityDefects(max_contour, hull)

        if defects is None:
            return -1, None

        points2 = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])

            angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

            if angle < 90:
                if start[1] < cy:
                    points2.append(start)

                if end[1] < cy:
                    points2.append(end)

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
            for point in points2:
                cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

        # STEP 6-3
        points = points1 + points2
        points = list(set(points))

        # STEP 6-4
        new_points = []
        for p0 in points:

            i = -1
            for index, c0 in enumerate(max_contour):
                c0 = tuple(c0[0])

                if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                    i = index
                    break

            if i >= 0:
                pre = i - 1
                if pre < 0:
                    pre = max_contour[len(max_contour) - 1][0]
                else:
                    pre = max_contour[i - 1][0]

                next = i + 1
                if next > len(max_contour) - 1:
                    next = max_contour[0][0]
                else:
                    next = max_contour[i + 1][0]

                if isinstance(pre, np.ndarray):
                    pre = tuple(pre.tolist())
                if isinstance(next, np.ndarray):
                    next = tuple(next.tolist())

                angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

                if angle < 90:
                    new_points.append(p0)

        return 1, new_points

    def process(img_bgr, debug):
        img_result = img_bgr.copy()

        # STEP 1
        img_bgr = removeFaceAra(img_bgr, cascade)

        # STEP 2
        img_binary = make_mask_image(img_bgr)

        # STEP 3
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
        cv.imshow("Binary", img_binary)

        # STEP 4
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if debug:
            for cnt in contours:
                cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

                # STEP 5
        max_area, max_contour = findMaxArea(contours)

        # if max_area == -1:
        # return img_result

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

            # STEP 6
        ret, points = getFingerPosition(max_contour, img_result, debug)

        finger_count = 0
        try:
            finger_count = len(points)
        except Exception as ex:
            print()
        # STEP 7
        # if ret > 0 and len(points) > 0:
        # for point in points:
        # cv.circle(img_result, point, 20, [255, 0, 255], 5)
        return finger_count

    current_file_path = os.path.dirname(os.path.realpath(__file__))
    cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))

    # cap = cv.VideoCapture('test.avi')

    cap = cv.VideoCapture(0)

    while True:

        ret, img_bgr = cap.read()

        if ret == False:
            break

        img_result = process(img_bgr, debug=False)

        key = cv.waitKey(1)
        if key == 27:
            break
        # cv.imshow("Result", img_result)

        if img_result == 1:
            return render(request, 'first/burger1.html')
        if img_result == 2:
            return render(request, 'first/burger2.html')
        if img_result == 3:
            return render(request, 'first/burger3.html')
        if img_result == 4:
            return render(request, 'first/burger4.html')
        if img_result == 5:
            return render(request, 'first/burger5.html')

        # if img_result == 1:
        #     return render(request, 'first/burger2.html')
        # if img_result == 2:
        #     return render(request, 'first/burger2.html')
        # if img_result == 3:
        #     return render(request, 'first/burger2.html')
        # if img_result == 4:
        #     return render(request, 'first/burger2.html')
        # if img_result == 5:
        #     return render(request, 'first/burger2.html')

    cap.release()
    cv.destroyAllWindows()
    # return render(request, 'first/profile.html')

def capture_fried(request):
    import cv2 as cv
    import numpy as np
    import os
    import pymysql

    def detect(img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    def removeFaceAra(img, cascade):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)

        height, width = img.shape[:2]

        for x1, y1, x2, y2 in rects:
            cv.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

        return img

    def make_mask_image(img_bgr):
        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

        # img_h,img_s,img_v = cv.split(img_hsv)

        low = (0, 30, 0)
        high = (15, 255, 255)

        img_mask = cv.inRange(img_hsv, low, high)
        return img_mask

    def distanceBetweenTwoPoints(start, end):
        x1, y1 = start
        x2, y2 = end

        return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))

    def calculateAngle(A, B):
        A_norm = np.linalg.norm(A)
        B_norm = np.linalg.norm(B)
        C = np.dot(A, B)

        angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
        return angle

    def findMaxArea(contours):
        max_contour = None
        max_area = -1

        for contour in contours:
            area = cv.contourArea(contour)

            x, y, w, h = cv.boundingRect(contour)

            if (w * h) * 0.4 > area:
                continue

            if w > h:
                continue

            if area > max_area:
                max_area = area
                max_contour = contour

        if max_area < 10000:
            max_area = -1

        return max_area, max_contour

    def getFingerPosition(max_contour, img_result, debug):
        points1 = []

        # STEP 6-1
        M = cv.moments(max_contour)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
        hull = cv.convexHull(max_contour)

        for point in hull:
            if cy > point[0][1]:
                points1.append(tuple(point[0]))

        if debug:
            cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
            for point in points1:
                cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

        # STEP 6-2
        hull = cv.convexHull(max_contour, returnPoints=False)
        defects = cv.convexityDefects(max_contour, hull)

        if defects is None:
            return -1, None

        points2 = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])

            angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

            if angle < 90:
                if start[1] < cy:
                    points2.append(start)

                if end[1] < cy:
                    points2.append(end)

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
            for point in points2:
                cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

        # STEP 6-3
        points = points1 + points2
        points = list(set(points))

        # STEP 6-4
        new_points = []
        for p0 in points:

            i = -1
            for index, c0 in enumerate(max_contour):
                c0 = tuple(c0[0])

                if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                    i = index
                    break

            if i >= 0:
                pre = i - 1
                if pre < 0:
                    pre = max_contour[len(max_contour) - 1][0]
                else:
                    pre = max_contour[i - 1][0]

                next = i + 1
                if next > len(max_contour) - 1:
                    next = max_contour[0][0]
                else:
                    next = max_contour[i + 1][0]

                if isinstance(pre, np.ndarray):
                    pre = tuple(pre.tolist())
                if isinstance(next, np.ndarray):
                    next = tuple(next.tolist())

                angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

                if angle < 90:
                    new_points.append(p0)

        return 1, new_points

    def process(img_bgr, debug):
        img_result = img_bgr.copy()

        # STEP 1
        img_bgr = removeFaceAra(img_bgr, cascade)

        # STEP 2
        img_binary = make_mask_image(img_bgr)

        # STEP 3
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
        cv.imshow("Binary", img_binary)

        # STEP 4
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if debug:
            for cnt in contours:
                cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

                # STEP 5
        max_area, max_contour = findMaxArea(contours)

        # if max_area == -1:
        # return img_result

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

            # STEP 6
        ret, points = getFingerPosition(max_contour, img_result, debug)

        finger_count = 0
        try:
            finger_count = len(points)
        except Exception as ex:
            print()
        # STEP 7
        # if ret > 0 and len(points) > 0:
        # for point in points:
        # cv.circle(img_result, point, 20, [255, 0, 255], 5)
        return finger_count

    current_file_path = os.path.dirname(os.path.realpath(__file__))
    cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))

    # cap = cv.VideoCapture('test.avi')

    cap = cv.VideoCapture(0)

    while True:

        ret, img_bgr = cap.read()

        if ret == False:
            break

        img_result = process(img_bgr, debug=False)

        key = cv.waitKey(1)
        if key == 27:
            break
        # cv.imshow("Result", img_result)

        if img_result == 1:
            return render(request, 'first/fried1.html')
        if img_result == 2:
            return render(request, 'first/fried2.html')
        if img_result == 3:
            return render(request, 'first/fried3.html')
        if img_result == 4:
            return render(request, 'first/fried4.html')
        if img_result == 5:
            return render(request, 'first/fried5.html')


    cap.release()
    cv.destroyAllWindows()
    # return render(request, 'first/profile.html')

def capture_coke(request):
    import cv2 as cv
    import numpy as np
    import os
    import pymysql

    def detect(img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    def removeFaceAra(img, cascade):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)

        height, width = img.shape[:2]

        for x1, y1, x2, y2 in rects:
            cv.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

        return img

    def make_mask_image(img_bgr):
        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

        # img_h,img_s,img_v = cv.split(img_hsv)

        low = (0, 30, 0)
        high = (15, 255, 255)

        img_mask = cv.inRange(img_hsv, low, high)
        return img_mask

    def distanceBetweenTwoPoints(start, end):
        x1, y1 = start
        x2, y2 = end

        return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))

    def calculateAngle(A, B):
        A_norm = np.linalg.norm(A)
        B_norm = np.linalg.norm(B)
        C = np.dot(A, B)

        angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
        return angle

    def findMaxArea(contours):
        max_contour = None
        max_area = -1

        for contour in contours:
            area = cv.contourArea(contour)

            x, y, w, h = cv.boundingRect(contour)

            if (w * h) * 0.4 > area:
                continue

            if w > h:
                continue

            if area > max_area:
                max_area = area
                max_contour = contour

        if max_area < 10000:
            max_area = -1

        return max_area, max_contour

    def getFingerPosition(max_contour, img_result, debug):
        points1 = []

        # STEP 6-1
        M = cv.moments(max_contour)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
        hull = cv.convexHull(max_contour)

        for point in hull:
            if cy > point[0][1]:
                points1.append(tuple(point[0]))

        if debug:
            cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
            for point in points1:
                cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

        # STEP 6-2
        hull = cv.convexHull(max_contour, returnPoints=False)
        defects = cv.convexityDefects(max_contour, hull)

        if defects is None:
            return -1, None

        points2 = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])

            angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

            if angle < 90:
                if start[1] < cy:
                    points2.append(start)

                if end[1] < cy:
                    points2.append(end)

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
            for point in points2:
                cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

        # STEP 6-3
        points = points1 + points2
        points = list(set(points))

        # STEP 6-4
        new_points = []
        for p0 in points:

            i = -1
            for index, c0 in enumerate(max_contour):
                c0 = tuple(c0[0])

                if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                    i = index
                    break

            if i >= 0:
                pre = i - 1
                if pre < 0:
                    pre = max_contour[len(max_contour) - 1][0]
                else:
                    pre = max_contour[i - 1][0]

                next = i + 1
                if next > len(max_contour) - 1:
                    next = max_contour[0][0]
                else:
                    next = max_contour[i + 1][0]

                if isinstance(pre, np.ndarray):
                    pre = tuple(pre.tolist())
                if isinstance(next, np.ndarray):
                    next = tuple(next.tolist())

                angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

                if angle < 90:
                    new_points.append(p0)

        return 1, new_points

    def process(img_bgr, debug):
        img_result = img_bgr.copy()

        # STEP 1
        img_bgr = removeFaceAra(img_bgr, cascade)

        # STEP 2
        img_binary = make_mask_image(img_bgr)

        # STEP 3
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
        cv.imshow("Binary", img_binary)

        # STEP 4
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if debug:
            for cnt in contours:
                cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

                # STEP 5
        max_area, max_contour = findMaxArea(contours)

        # if max_area == -1:
        # return img_result

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

            # STEP 6
        ret, points = getFingerPosition(max_contour, img_result, debug)

        finger_count = 0
        try:
            finger_count = len(points)
        except Exception as ex:
            print()
        # STEP 7
        # if ret > 0 and len(points) > 0:
        # for point in points:
        # cv.circle(img_result, point, 20, [255, 0, 255], 5)
        return finger_count

    current_file_path = os.path.dirname(os.path.realpath(__file__))
    cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))

    # cap = cv.VideoCapture('test.avi')

    cap = cv.VideoCapture(0)

    while True:

        ret, img_bgr = cap.read()

        if ret == False:
            break

        img_result = process(img_bgr, debug=False)

        key = cv.waitKey(1)
        if key == 27:
            break
        # cv.imshow("Result", img_result)

        # if img_result == 1:
        #     return render(request, 'first/coke1.html')
        # if img_result == 2:
        #     return render(request, 'first/coke2.html')
        # if img_result == 3:
        #     return render(request, 'first/coke3.html')
        # if img_result == 4:
        #     return render(request, 'first/coke4.html')
        # if img_result == 5:
        #     return render(request, 'first/coke5.html')

        if img_result == 1:
            return render(request, 'first/coke1.html')
        if img_result == 2:
            return render(request, 'first/coke1.html')
        if img_result == 3:
            return render(request, 'first/coke1.html')
        if img_result == 4:
            return render(request, 'first/coke1.html')
        if img_result == 5:
            return render(request, 'first/coke1.html')


    cap.release()
    cv.destroyAllWindows()
    # return render(request, 'first/profile.html')

def capture_ice(request):
    import cv2 as cv
    import numpy as np
    import os
    import pymysql

    def detect(img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    def removeFaceAra(img, cascade):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)

        height, width = img.shape[:2]

        for x1, y1, x2, y2 in rects:
            cv.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

        return img

    def make_mask_image(img_bgr):
        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

        # img_h,img_s,img_v = cv.split(img_hsv)

        low = (0, 30, 0)
        high = (15, 255, 255)

        img_mask = cv.inRange(img_hsv, low, high)
        return img_mask

    def distanceBetweenTwoPoints(start, end):
        x1, y1 = start
        x2, y2 = end

        return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))

    def calculateAngle(A, B):
        A_norm = np.linalg.norm(A)
        B_norm = np.linalg.norm(B)
        C = np.dot(A, B)

        angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
        return angle

    def findMaxArea(contours):
        max_contour = None
        max_area = -1

        for contour in contours:
            area = cv.contourArea(contour)

            x, y, w, h = cv.boundingRect(contour)

            if (w * h) * 0.4 > area:
                continue

            if w > h:
                continue

            if area > max_area:
                max_area = area
                max_contour = contour

        if max_area < 10000:
            max_area = -1

        return max_area, max_contour

    def getFingerPosition(max_contour, img_result, debug):
        points1 = []

        # STEP 6-1
        M = cv.moments(max_contour)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
        hull = cv.convexHull(max_contour)

        for point in hull:
            if cy > point[0][1]:
                points1.append(tuple(point[0]))

        if debug:
            cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
            for point in points1:
                cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

        # STEP 6-2
        hull = cv.convexHull(max_contour, returnPoints=False)
        defects = cv.convexityDefects(max_contour, hull)

        if defects is None:
            return -1, None

        points2 = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])

            angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

            if angle < 90:
                if start[1] < cy:
                    points2.append(start)

                if end[1] < cy:
                    points2.append(end)

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
            for point in points2:
                cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

        # STEP 6-3
        points = points1 + points2
        points = list(set(points))

        # STEP 6-4
        new_points = []
        for p0 in points:

            i = -1
            for index, c0 in enumerate(max_contour):
                c0 = tuple(c0[0])

                if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                    i = index
                    break

            if i >= 0:
                pre = i - 1
                if pre < 0:
                    pre = max_contour[len(max_contour) - 1][0]
                else:
                    pre = max_contour[i - 1][0]

                next = i + 1
                if next > len(max_contour) - 1:
                    next = max_contour[0][0]
                else:
                    next = max_contour[i + 1][0]

                if isinstance(pre, np.ndarray):
                    pre = tuple(pre.tolist())
                if isinstance(next, np.ndarray):
                    next = tuple(next.tolist())

                angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

                if angle < 90:
                    new_points.append(p0)

        return 1, new_points

    def process(img_bgr, debug):
        img_result = img_bgr.copy()

        # STEP 1
        img_bgr = removeFaceAra(img_bgr, cascade)

        # STEP 2
        img_binary = make_mask_image(img_bgr)

        # STEP 3
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
        cv.imshow("Binary", img_binary)

        # STEP 4
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if debug:
            for cnt in contours:
                cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

                # STEP 5
        max_area, max_contour = findMaxArea(contours)

        # if max_area == -1:
        # return img_result

        if debug:
            cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

            # STEP 6
        ret, points = getFingerPosition(max_contour, img_result, debug)

        finger_count = 0
        try:
            finger_count = len(points)
        except Exception as ex:
            print()
        # STEP 7
        # if ret > 0 and len(points) > 0:
        # for point in points:
        # cv.circle(img_result, point, 20, [255, 0, 255], 5)
        return finger_count

    current_file_path = os.path.dirname(os.path.realpath(__file__))
    cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))

    # cap = cv.VideoCapture('test.avi')

    cap = cv.VideoCapture(0)

    while True:

        ret, img_bgr = cap.read()

        if ret == False:
            break

        img_result = process(img_bgr, debug=False)

        key = cv.waitKey(1)
        if key == 27:
            break
        # cv.imshow("Result", img_result)

        if img_result == 1:
            return render(request, 'first/ice1.html')
        if img_result == 2:
            return render(request, 'first/ice2.html')
        if img_result == 3:
            return render(request, 'first/ice3.html')
        if img_result == 4:
            return render(request, 'first/ice4.html')
        if img_result == 5:
            return render(request, 'first/ice5.html')


    cap.release()
    cv.destroyAllWindows()
    # return render(request, 'first/profile.html')

def burger1(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '1' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger1check(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '1' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger2(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '2' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger2check(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '2' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger3(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '3' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger3check(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '3' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger4(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '4' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger4check(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '4' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger5(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '5' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def burger5check(request):
    persons = Person.objects.filter(id=3800)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '5' WHERE (`ID` = '3800')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried1(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '1' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried1check(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '1' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried2(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '2' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried2check(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '2' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried3(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '3' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried3check(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '3' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried4(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '4' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried4check(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '4' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried5(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '5' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def fried5check(request):
    persons = Person.objects.filter(id=1700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '5' WHERE (`ID` = '1700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke1(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '1' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke1check(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '1' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke2(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '2' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke2check(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '2' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke3(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '3' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke3check(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '3' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke4(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '4' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke4check(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '4' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke5(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '5' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def coke5check(request):
    persons = Person.objects.filter(id=1500)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '5' WHERE (`ID` = '1500')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice1(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '1' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice1check(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '1' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice2(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '2' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice2check(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '2' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice3(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '3' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice3check(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '3' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice4(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '4' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice4check(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '4' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice5(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '5' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def ice5check(request):
    persons = Person.objects.filter(id=700)
    try:
        cursor = connection.cursor()

        sql = "UPDATE `kioskdata`.`person` SET `Phone` = '5' WHERE (`ID` = '700')"
        result = cursor.execute(sql)
        datas = cursor.fetchall()

        connection.commit()
        connection.close()
    except:
        connection.rollback()
        print("Failed")

    context = {'persons': persons}
    return render(request, 'first/kiosk_menu.html', context)

def third_page(request):
    personsbs = Person.objects.filter(id=3800)
    personsfs = Person.objects.filter(id=1700)
    personscs = Person.objects.filter(id=1500)
    personsis = Person.objects.filter(id=700)
    personsalls = Person.objects.all()

    context = {'personsbs': personsbs, 'personsfs': personsfs, 'personscs': personscs, 'personsis': personsis, 'personsalls': personsalls}
    return render(request, 'first/third_page.html', context)

def four_page(request):
    return render(request, 'first/four_page.html')

def paycomplete(request):
    return render(request, 'first/paycomplete.html')

def multiprocessing(request):
    capture(request)
    return render(request, 'first/page-profile.html')

def myinfo(request):
    todaym6 = DateFormat(datetime.now() - timedelta(6)).format('Y-m-d')
    todaym5 = DateFormat(datetime.now() - timedelta(5)).format('Y-m-d')
    todaym4 = DateFormat(datetime.now() - timedelta(4)).format('Y-m-d')
    todaym3 = DateFormat(datetime.now() - timedelta(3)).format('Y-m-d')
    todaym2 = DateFormat(datetime.now() - timedelta(2)).format('Y-m-d')
    todaym1 = DateFormat(datetime.now() - timedelta(1)).format('Y-m-d')
    today = DateFormat(datetime.now()).format('Y-m-d')
    staticss = Statics.objects.all()
    static01s = Statics.objects.filter(day=todaym6)
    static02s = Statics.objects.filter(day=todaym5)
    static03s = Statics.objects.filter(day=todaym4)
    static04s = Statics.objects.filter(day=todaym3)
    static05s = Statics.objects.filter(day=todaym2)
    static06s = Statics.objects.filter(day=todaym1)
    static07s = Statics.objects.filter(day=today)
    times = Kioskaccess.objects.all()[:5]
    states = State.objects.all()[:5]

    context = {'times': times, 'states':states, 'staticss': staticss, 'static01s': static01s, 'static02s': static02s,
               'static03s': static03s, 'static04s': static04s, 'static05s': static05s, 'static06s': static06s, 'static07s': static07s}
    return render(request, 'first/myinfo.html', context)

def charts(request):
    todaym29 = DateFormat(datetime.now() - timedelta(29)).format('Y-m-d')
    todaym28 = DateFormat(datetime.now() - timedelta(28)).format('Y-m-d')
    todaym27 = DateFormat(datetime.now() - timedelta(27)).format('Y-m-d')
    todaym26 = DateFormat(datetime.now() - timedelta(26)).format('Y-m-d')
    todaym25 = DateFormat(datetime.now() - timedelta(25)).format('Y-m-d')
    todaym24 = DateFormat(datetime.now() - timedelta(24)).format('Y-m-d')
    todaym23 = DateFormat(datetime.now() - timedelta(23)).format('Y-m-d')
    todaym22 = DateFormat(datetime.now() - timedelta(22)).format('Y-m-d')
    todaym21 = DateFormat(datetime.now() - timedelta(21)).format('Y-m-d')
    todaym20 = DateFormat(datetime.now() - timedelta(20)).format('Y-m-d')
    todaym19 = DateFormat(datetime.now() - timedelta(19)).format('Y-m-d')
    todaym18 = DateFormat(datetime.now() - timedelta(18)).format('Y-m-d')
    todaym17 = DateFormat(datetime.now() - timedelta(17)).format('Y-m-d')
    todaym16 = DateFormat(datetime.now() - timedelta(16)).format('Y-m-d')
    todaym15 = DateFormat(datetime.now() - timedelta(15)).format('Y-m-d')
    todaym14 = DateFormat(datetime.now() - timedelta(14)).format('Y-m-d')
    todaym13 = DateFormat(datetime.now() - timedelta(13)).format('Y-m-d')
    todaym12 = DateFormat(datetime.now() - timedelta(12)).format('Y-m-d')
    todaym11 = DateFormat(datetime.now() - timedelta(11)).format('Y-m-d')
    todaym10 = DateFormat(datetime.now() - timedelta(10)).format('Y-m-d')
    todaym9 = DateFormat(datetime.now() - timedelta(9)).format('Y-m-d')
    todaym8 = DateFormat(datetime.now() - timedelta(8)).format('Y-m-d')
    todaym7 = DateFormat(datetime.now() - timedelta(7)).format('Y-m-d')
    todaym6 = DateFormat(datetime.now() - timedelta(6)).format('Y-m-d')
    todaym5 = DateFormat(datetime.now() - timedelta(5)).format('Y-m-d')
    todaym4 = DateFormat(datetime.now() - timedelta(4)).format('Y-m-d')
    todaym3 = DateFormat(datetime.now() - timedelta(3)).format('Y-m-d')
    todaym2 = DateFormat(datetime.now() - timedelta(2)).format('Y-m-d')
    todaym1 = DateFormat(datetime.now() - timedelta(1)).format('Y-m-d')
    today = DateFormat(datetime.now()).format('Y-m-d')
    staticss = Statics.objects.all()
    static01s = Statics.objects.filter(day=todaym29)
    static02s = Statics.objects.filter(day=todaym28)
    static03s = Statics.objects.filter(day=todaym27)
    static04s = Statics.objects.filter(day=todaym26)
    static05s = Statics.objects.filter(day=todaym25)
    static06s = Statics.objects.filter(day=todaym24)
    static07s = Statics.objects.filter(day=todaym23)
    static08s = Statics.objects.filter(day=todaym22)
    static09s = Statics.objects.filter(day=todaym21)
    static10s = Statics.objects.filter(day=todaym20)
    static11s = Statics.objects.filter(day=todaym19)
    static12s = Statics.objects.filter(day=todaym18)
    static13s = Statics.objects.filter(day=todaym17)
    static14s = Statics.objects.filter(day=todaym16)
    static15s = Statics.objects.filter(day=todaym15)
    static16s = Statics.objects.filter(day=todaym14)
    static17s = Statics.objects.filter(day=todaym13)
    static18s = Statics.objects.filter(day=todaym12)
    static19s = Statics.objects.filter(day=todaym11)
    static20s = Statics.objects.filter(day=todaym10)
    static21s = Statics.objects.filter(day=todaym9)
    static22s = Statics.objects.filter(day=todaym8)
    static23s = Statics.objects.filter(day=todaym7)
    static24s = Statics.objects.filter(day=todaym6)
    static25s = Statics.objects.filter(day=todaym5)
    static26s = Statics.objects.filter(day=todaym4)
    static27s = Statics.objects.filter(day=todaym3)
    static28s = Statics.objects.filter(day=todaym2)
    static29s = Statics.objects.filter(day=todaym1)
    static30s = Statics.objects.filter(day=today)
    context = {'staticss': staticss, 'static01s': static01s, 'static02s': static02s, 'static03s': static03s,
               'static04s': static04s, 'static05s': static05s, 'static06s': static06s, 'static07s': static07s,
                'static08s': static08s, 'static09s': static09s, 'static10s': static10s,
               'static11s': static11s, 'static12s': static12s, 'static13s': static13s,
               'static14s': static14s, 'static15s': static15s, 'static16s': static16s, 'static17s': static17s,
               'static18s': static18s, 'static19s': static19s, 'static20s': static20s,
               'static21s': static21s, 'static22s': static22s, 'static23s': static23s,
               'static24s': static24s, 'static25s': static25s, 'static26s': static26s, 'static27s': static27s,
               'static28s': static28s, 'static29s': static29s, 'static30s': static30s,
               }
    return render(request, 'first/charts.html', context)


def tables(request):
    kstates = Kstate.objects.all()#table1에 필요
    dailylist = Kiosk.objects.all() #daily access list 확장판
    recent_kioskaccess = Kioskaccess.objects.all().order_by('timedata') #recently time 순서대로 출력해야함
    symptom = State.objects.all()
    ob = Statics.objects.filter(day='2020-11-01')
    persons_info = Person.objects.all()


    context = {'kstates': kstates, 'dailylist': dailylist, 'recent_kioskaccess': recent_kioskaccess, 'symptom': symptom, 'ob':ob, 'persons_info':persons_info}
    return render(request, 'first/tables.html', context)

def login(request):
    try:
        request.session['userid']
        return HttpResponseRedirect(reverse('first/login'))
    except:
        return render(request, 'first/login.html')


def sign_in(request):
    user_id = request.POST['user_id']
    user_pw = request.POST['user_pw']
    try:
        user = User.objects.get(userid=user_id, password=user_pw)
        request.session['userid'] = user_id
        return HttpResponseRedirect(reverse('myinfo'))
    except User.DoesNotExist:
        return HttpResponse('실패')


def sign_up(request):
    return render(request, 'first/signup.html')

def join(request):
    user_id = request.POST['id']
    user_pw = request.POST['pw']
    user_name = request.POST['name']
    try:
        user = User.objects.get(userid=user_id)
        return HttpResponse('이미 존재하는 아이디')
    except User.DoesNotExist:
        user = User(userid=user_id, password=user_pw, username=user_name)
        user.save()

        keywords = SubjectKeyword.objects.all()
        for keyword in keywords:
            UserKeyword.objects.update_or_create(
                user_id=user.id,
                keyword_id=keyword.keyword_id,
                keyword=keyword.keyword,
                flag=0
            )
        request.session['userid'] = user_id
        return HttpResponseRedirect(reverse('index'))

    return HttpResponseRedirect(reverse('first/signup'))



# def myinfo(request):
#     keyword_list = []
#     try:
#         user = User.objects.get(userid=request.session['userid'])
#         user_keyword = []
#         try:
#             keywords = SubjectKeyword.objects.all()
#             try:
#                 user_keyword = get_list_or_404(UserKeyword, user_id=user.id)
#             except:
#                 pass
#             """
#             page = int(request.GET.get('p',1))
#             paginator = Paginator(keywords,5)
#             queryset1 = paginator.get_page(page)
#             try:
#                 contacts = paginator.page(page)
#             except PageNotAnInteger:
#                 contacts = paginator.page(1)
#             except EmptyPage:
#                 contacts = paginator.page(paginator.num_pages)
#             """
#             paginator = Paginator(keywords, 5)
#
#             page_number = request.GET.get('page')
#             page_obj = paginator.get_page(page_number)
#
#             keywords = SubjectKeyword.objects.all()
#             if len(keywords) > len(user_keyword):
#                 for keyword in keywords:
#                     try:
#                         UserKeyword.objects.get(user_id=user.id, keyword_id=keyword.keyword_id)
#                     except:
#                         UserKeyword.objects.update_or_create(
#                             user_id=user.id,
#                             keyword_id=keyword.keyword_id,
#                             keyword=keyword.keyword,
#                             flag=0
#                         )
#                 user_keyword = get_list_or_404(UserKeyword, user_id=user.id)
#         except:
#             pass
#         return render(request, 'first/myinfo.html', {'user': get_object_or_404(User, userid=request.session['userid']),
#                                                     'user_keywords': user_keyword, 'page_obj': page_obj})
#     except KeyError:
#         return HttpResponseRedirect(reverse('index'))

def keyword(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        user_keyword = UserKeyword.objects.get(user_id=user.id, keyword_id=id)
        if user_keyword.flag == 0:
            user_keyword.flag = 1
        else:
            user_keyword.flag = 0
        user_keyword.save()
        return HttpResponseRedirect(reverse('myinfo'))
    except KeyError:
        return HttpResponseRedirect(reverse('myinfo'))


def page_profile(request):
    Statss1 = State.objects.filter(id__address='청주시 서원구')  # table1에 필요
    Statss2 = State.objects.filter(id__address='청주시 흥덕구')
    Statss3 = State.objects.filter(id__address='청주시 상당구')
    Statss4 = State.objects.filter(id__address='청주시 청원구')
    Persons_ = Person.objects.all()  # daily access list 확장판
    Address_ = pd.DataFrame(Person.objects.values('address'))
    states = State.objects.all()
    #stateperson = State.objects.select_related(name='흥덕구%').all()

    # >> > brewed_coffee_id = Category.objects.get(name="브루드 커피")
    # >> > brewed_coffe_list = Drink.objects.filter(category_id=brewed_coffee_id)
    # >> > for coffee in brewed_coffe_list:
    #     ...
    #     print(coffee.name)
    statequery = State.objects.select_related("id").all()
    # state_list = []
    # for list in statequery:
    #     state_list.append({
    #         'suspect' : list.suspect,
    #         'confirm' : list.confirm,
    #         'id' : list.id,
    #     })

    # queryset = Book.objects.select_related("publisher").all()
    name_split = Address_["address"].str.split(" ")

    Address_["first_name"] = name_split.str.get(0)
    Address_["GWOO"] = name_split.str.get(1)
    # 서원구로  person id 로 연결 id로 가서 사람 체크
    # 서원구로 person->id 로 개수 세기
    hdcount = 0
    swcount = 0
    sdcount = 0
    chwcount =0
    for address in Address_["GWOO"]:
        if address == "흥덕구":
            hdcount = hdcount + 1
        elif address == "서원구":
            swcount = swcount + 1
        elif address == "상당구":
            sdcount = sdcount + 1
        elif address == "청원구":
            chwcount = chwcount + 1
    print(hdcount)
    print(swcount)
    print(sdcount)
    print(chwcount)
    staticslist = Statics.objects.all()
    context = { 'hdcount' : hdcount,
             'swcount' : swcount,
             'sdcount' : sdcount,
             'chwcount' : chwcount,
            'staticslist': staticslist,
                'Persons_': Persons_,
                'Statss1':Statss1,
                'Statss2': Statss2,
                'Statss3': Statss3,
                'Statss4': Statss4,
                'states': states
                }

    return render(request, 'first/page-profile.html', context)