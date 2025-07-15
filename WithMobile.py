import cv2
import mediapipe as mp
import pyttsx3
import threading
import numpy as np

# Voice setup
engine = pyttsx3.init()
def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

# Load and resize tool icons (increased width to 400, height remains 200)
tools = cv2.imread("tools.png")
tools = cv2.resize(tools, (400, 200))  # New width: 400
tools = tools.astype("uint8")

ml = 150
max_x, max_y = ml + 400, 200  # Updated max_x to match new tool width
tool_names = ["Pen", "Rectangle", "Curve", "Circle", "Eraser"]
tool_regions = [(ml + i * 80, 0, ml + (i + 1) * 80, max_y) for i in range(5)]  # Each tool 80px wide now
tool_dimensions = {}

# States
prev_tool = None
selected_tool = None
active_tool = None
hover_counter = 0
waiting_for_input = False
input_text = ""
input_prompt = ""
current_dimension_key = ""
prev_point = None
stable_counter = 0

# Drawing flag
is_drawing = False

# Use IP Webcam
ip_webcam_url = "http://192.168.130.60:8080/video"  # Replace with your own
cap = cv2.VideoCapture(ip_webcam_url)

if not cap.isOpened():
    print("Error: Unable to access IP webcam stream. Check the URL and WiFi connection.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from IP Webcam. Check connection.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_pos = None
    is_two_fingers = False

    # Safely overlay tools
    tool_area = frame[0:max_y, ml:max_x]
    resized_tools = cv2.resize(tools, (tool_area.shape[1], tool_area.shape[0]))
    frame[0:max_y, ml:max_x] = cv2.addWeighted(resized_tools, 0.7, tool_area, 0.3, 0)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            index_base = hand_landmarks.landmark[6]
            middle_base = hand_landmarks.landmark[10]

            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)
            finger_pos = (x, y)

            if index_tip.y < index_base.y and middle_tip.y < middle_base.y:
                is_two_fingers = True
            else:
                is_two_fingers = False

            if not waiting_for_input and not is_two_fingers:
                hovered_tool = None
                for i, (x1, y1, x2, y2) in enumerate(tool_regions):
                    if x1 < x < x2 and y1 < y < y2:
                        hovered_tool = tool_names[i]
                        break

                if hovered_tool:
                    if hovered_tool == prev_tool:
                        hover_counter += 1
                    else:
                        hover_counter = 1
                        prev_tool = hovered_tool

                    if hover_counter == 20:
                        selected_tool = hovered_tool
                        speak(f"{selected_tool} selected")
                        input_text = ""
                        current_dimension_key = selected_tool
                        waiting_for_input = True
                        if selected_tool == "Circle":
                            input_prompt = "Enter Radius:"
                        elif selected_tool == "Rectangle":
                            input_prompt = "Enter Width,Height:"
                        elif selected_tool == "Pen":
                            input_prompt = "Enter Pen Size:"
                        elif selected_tool == "Eraser":
                            input_prompt = "Enter Eraser Size:"
                        elif selected_tool == "Curve":
                            input_prompt = "Enter Curve Size:"
                else:
                    hover_counter = 0
                    prev_tool = None

            cv2.circle(frame, (x, y), 10, (0, 0, 255) if is_two_fingers else (255, 0, 0), -1)

    if finger_pos and not waiting_for_input and active_tool in tool_dimensions and is_two_fingers:
        is_drawing = True
        if active_tool in ["Pen", "Eraser", "Curve"]:
            color = (0, 0, 0) if active_tool == "Pen" else (255, 255, 255)
            if active_tool == "Curve":
                color = (150, 0, 200)
            size = int(tool_dimensions[active_tool])
            if prev_point:
                cv2.line(canvas, prev_point, finger_pos, color, size)
            prev_point = finger_pos

        elif active_tool == "Circle":
            stable_counter += 1
            if stable_counter >= 10:
                radius = int(tool_dimensions["Circle"])
                cv2.circle(canvas, finger_pos, radius, (0, 255, 0), 2)
                speak("Circle drawn")
                active_tool = None
                stable_counter = 0
                prev_point = None

        elif active_tool == "Rectangle":
            stable_counter += 1
            if stable_counter >= 10:
                try:
                    w, h = map(int, tool_dimensions["Rectangle"].split(","))
                    top_left = finger_pos
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(canvas, top_left, bottom_right, (255, 0, 255), 2)
                    speak("Rectangle drawn")
                    active_tool = None
                except:
                    speak("Invalid dimensions")
                stable_counter = 0
                prev_point = None
    else:
        prev_point = None
        stable_counter = 0
        is_drawing = False

    display_frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    if selected_tool:
        cv2.putText(display_frame, f'Tool: {selected_tool}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if selected_tool in tool_dimensions:
            cv2.putText(display_frame, f'{tool_dimensions[selected_tool]}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    if waiting_for_input:
        cv2.rectangle(display_frame, (50, 150), (600, 220), (255, 255, 255), -1)
        cv2.putText(display_frame, input_prompt, (60, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(display_frame, input_text, (60, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(display_frame, f'Drawing: {"Yes" if is_drawing else "No"}', (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

    cv2.imshow("Hand Drawing", display_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if waiting_for_input:
        if key in [13, 10]:  # Enter
            tool_dimensions[current_dimension_key] = input_text
            speak(f"{current_dimension_key} set to {input_text}")
            waiting_for_input = False
            active_tool = selected_tool
        elif key == 8:  # Backspace
            input_text = input_text[:-1]
        elif key != -1 and 32 <= key <= 126:
            input_text += chr(key)

cap.release()
cv2.destroyAllWindows()