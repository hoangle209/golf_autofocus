import cv2
import os
import argparse

from golf_autofocus.utils import (
                        calc_laplacian_var,
                        calc_brightness,
                        calc_blurriness,
                        calc_entropy
)

def read_conifg():
    if os.path.exists("config.txt"):
        with open("config.txt", 'r') as f:
            data = f.read().split()
        var_ref = float(data[0])
        brightness = float(data[1])
        darkness = float(data[2])
        blurriness = float(data[3])
        entropy = float(data[4])
        points = list(map(int, data[5:]))

        return var_ref, points, brightness, darkness, blurriness, entropy
    
    return None, [], None, None, None, None

def visualize(PATH):
    var, points, brightness, darkness, blurriness, entropy = read_conifg()
    var_ref = var
    brightness_ref = brightness
    darkness_ref = darkness
    blurriness_ref = blurriness
    entropy_ref = entropy

    def handle_mouse_events(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN: 
            if len(points) < 8:
                points += [x, y]
        if event == cv2.EVENT_RBUTTONDOWN: 
            cv2.destroyAllWindows()
            exit()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("image", 960, 816) 
    cv2.setMouseCallback("image", handle_mouse_events)

    cap = cv2.VideoCapture(PATH)

    while True:
        suc, frame = cap.read()
        if not suc:
            break

        while True:
            frame_copy = frame.copy()
            
            if len(points)==8:
                xmin = min(points[::2])
                xmax = max(points[::2])
                ymin = min(points[1::2])
                ymax = max(points[1::2])

                crop = frame_copy[ymin:ymax, xmin:xmax]

                var = calc_laplacian_var(crop)
                brightness, darkness = calc_brightness(crop)
                blurriness = calc_blurriness(crop)
                entropy = calc_entropy(crop)
                
                r_var = var/var_ref if var < var_ref else var_ref/var
                r_brightness = brightness/brightness_ref if brightness < brightness_ref else brightness_ref/brightness
                r_darkness = darkness/darkness_ref if darkness < darkness_ref else darkness_ref/darkness
                r_blurriness = blurriness/blurriness_ref if blurriness < blurriness_ref else blurriness_ref/blurriness
                r_entropy = entropy/entropy_ref if entropy < entropy_ref else entropy_ref/entropy
                accept = int(r_var > 0.95) + \
                         int(r_brightness > 0.9) + \
                         int(r_darkness > 0.9) + \
                         int(r_entropy > 0.9) + \
                         int(r_blurriness > 0.9)
                
                if accept > 3:
                    cv2.putText(frame_copy, f"OK", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                3, (0,255,0), 2, cv2.LINE_AA)
                    c = (0,255,0)
                else:
                    cv2.putText(frame_copy, f"Not OK", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                3, (0,0,255), 2, cv2.LINE_AA)
                    c = (0,0,255)

                cv2.putText(frame_copy, f"{var}", (xmin, ymin-100), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, c, 2, cv2.LINE_AA)
                
                cv2.putText(frame_copy, f"{brightness:0.4f} - {darkness:0.4f}", (xmin, ymin-60), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, c, 2, cv2.LINE_AA)
                cv2.putText(frame_copy, f"{blurriness:0.4f} - {entropy:0.4f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, c, 2, cv2.LINE_AA)
                
                cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), c, 3)

            cv2.imshow("image", frame_copy)

            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break
            elif key == ord('q') or key==ord('Q'):
                points = []
            # write data to config.txt
            elif key == ord('b') or key==ord('B'):
                var_ref = var
                with open("config.txt", 'w') as f:
                    if len(points) == 8 and var is not None:
                        f.write(str(var)) # laplacian variance
                        f.write(" ")
                        f.write(str(brightness)) # brightness
                        f.write(" ")
                        f.write(str(darkness)) # darkness
                        f.write(" ")
                        f.write(str(blurriness)) # blurriness
                        f.write(" ")
                        f.write(str(entropy)) # entropy
                        f.write(" ")
                        f.write(" ".join(list(map(str, points)))) # bounding boxes

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MQ Auto Focus")
    parser.add_argument("-p", "--path", type=str, help="Video Path")

    # PATH = "vlc-record-2024-09-09-13h14m09s-v4l2____dev_video0-.avi"
    args = parser.parse_args()
    PATH = args.path

    visualize(PATH)


