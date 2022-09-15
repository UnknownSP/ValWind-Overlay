from ast import Or
from asyncio.windows_events import NULL
import enum
import os
from re import A
import time
from turtle import left

import cv2
import numpy as np
import socketio
import win32con
import win32gui
import win32ui

#window = win32gui.FindWindow(None, "VALORANT  ")
#if window == NULL or window == 0:
#    print("No Window Found")
#    exit()
#x0, y0, x1, y1 = win32gui.GetWindowRect(window)
#width = x1 - x0
#height = y1 - y0
#
#window_DC = win32gui.GetWindowDC(window)
#DC_Obj = win32ui.CreateDCFromHandle(window_DC)
#com_DC = DC_Obj.CreateCompatibleDC()
#dataBitMap = win32ui.CreateBitmap()
#dataBitMap.CreateCompatibleBitmap(DC_Obj, width, height)
#com_DC.SelectObject(dataBitMap)
#com_DC.BitBlt((0, 0), (width, height), DC_Obj, (0, 0), win32con.SRCCOPY)
#
#bmpstr = dataBitMap.GetBitmapBits(True)
#
#img = np.frombuffer(dataBitMap.GetBitmapBits(True), np.uint8).reshape(height, width, 4)
#img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#
#DC_Obj.DeleteDC()
#com_DC.DeleteDC()
#win32gui.ReleaseDC(window, window_DC)
#win32gui.DeleteObject(dataBitMap.GetHandle())
#
#cv2.imshow("", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(window)

AGENTS_NAME = ["Fade","Breach","Raze","Chamber","KAYO","Skye","Cypher","Sova","Killjoy","Viper","Phoenix","Astra","Brimstone","Neon","Yoru","Sage","Reyna","Omen","Jett"]

def WindowCapture(window_name: str, bgr2rgb: bool = False):
    # 現在アクティブなウィンドウ名を探す
    process_list = []

    def callback(handle, _):
        process_list.append(win32gui.GetWindowText(handle))
    win32gui.EnumWindows(callback, None)

    # ターゲットウィンドウ名を探す
    for process_name in process_list:
        if window_name in process_name:
            hnd = win32gui.FindWindow(None, process_name)
            break
    else:
        # 見つからなかったら画面全体を取得
        hnd = win32gui.GetDesktopWindow()

    while True:
        start = time.time()
        # ウィンドウサイズ取得
        x0, y0, x1, y1 = win32gui.GetWindowRect(hnd)
        width = x1 - x0
        height = y1 - y0
        # ウィンドウのデバイスコンテキスト取得
        windc = win32gui.GetWindowDC(hnd)
        srcdc = win32ui.CreateDCFromHandle(windc)
        memdc = srcdc.CreateCompatibleDC()
        # デバイスコンテキストからピクセル情報コピー, bmp化
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)
        img = np.frombuffer(bmp.GetBitmapBits(True), np.uint8).reshape(height, width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = img[60 : 1140, 0 : 1920]
        end = time.time()
        print(end-start)
        if end-start < 1.0:
            while True:
                if(time.time()-start >= 0.999):
                    break
    
        cv2.destroyAllWindows()
        cv2.imshow("image", img)
        cv2.waitKey(1)
        # 後片付け
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hnd, windc)
        win32gui.DeleteObject(bmp.GetHandle())

    #return img
def testImageCapture():
    img = cv2.imread("./test_image/main4.png")
    img = img[60 : 1140, 0 : 1920]
    left_play_agents = ["Phoenix","Raze","Reyna","Sage","Yoru"]
    right_play_agents = ["Phoenix","Raze","Reyna","Sage","Yoru"]
    left_agents_template, right_agents_template = generate_templates(left_play_agents, right_play_agents)
    left_live_agents_raw, right_live_agents_raw = get_LiveAgentsFrame(img)
    get_LiveAgents(left_agents_template,left_live_agents_raw,left_play_agents,right_agents_template,right_live_agents_raw,right_play_agents)
    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def get_LiveAgents(left_template,left_checkImg,left_agents,right_template,right_checkImg,right_agents):
    left_live_agents = []
    for i,check_image in enumerate(left_checkImg):
        live = False
        for j,template in enumerate(left_template):
            result = cv2.matchTemplate(check_image, template, cv2.TM_CCOEFF_NORMED)
            max_location = cv2.minMaxLoc(result)[1]
            if max_location > 0.7:
                left_live_agents.append(left_agents[j])
                live = True
                break
        if not live:
            left_live_agents.append(NULL)
    print(left_live_agents)
                


def generate_templates(left_agents_names, right_agents_names):
    left_template = []
    right_template = []
    for agents_name in left_agents_names:
        template = cv2.imread("./templates/header_agents/" + agents_name + ".png")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        left_template.append(template)
    for agents_name in right_agents_names:
        template = cv2.imread("./templates/header_agents/" + agents_name + ".png")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.flip(template,1)
        right_template.append(template)
    return left_template, right_template


def get_LiveAgentsFrame(image):
    left_agents = []
    right_agents = []
    y_higher = 70
    y_lower = 30
    x_between_agent_width = 66
    x_agent_width = 40
    x_start = 446
    x_end = x_start + x_agent_width
    for i in range(0,5):
        cropped_agent = image[y_lower:y_higher,x_start:x_end]
        cropped_agent = cv2.cvtColor(cropped_agent, cv2.COLOR_BGR2GRAY)
        left_agents.append(cropped_agent)
        x_start = x_start + x_between_agent_width
        x_end = x_start + x_agent_width
    
    x_start = 1171
    x_end = x_start + x_agent_width
    for i in range(0,5):
        cropped_agent = image[y_lower:y_higher,x_start:x_end]
        cropped_agent = cv2.cvtColor(cropped_agent, cv2.COLOR_BGR2GRAY)
        right_agents.append(cropped_agent)
        x_start = x_start + x_between_agent_width
        x_end = x_start + x_agent_width
    
    return left_agents, right_agents

    for check_image in right_agents:
        result = cv2.matchTemplate(check_image, template_flip, cv2.TM_CCOEFF_NORMED)
        max_location = cv2.minMaxLoc(result)[1]
        print(max_location)
    #for i in range(0,5):
    #    cv2.imshow("image"+str(i),all_agents[i])
    #    cv2.imwrite("image"+str(i)+".png",all_agents[i])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#WindowCapture("??????") # 部分一致
testImageCapture()