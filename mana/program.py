import cv2
import numpy as np
from matplotlib.patches import Rectangle
import os
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

global displayed_image

def main():
    root = tk.Tk()
    root.title("Emoji Finder")
    root.geometry("789x742")
    image_path = tk.StringVar()
    displayed_image = None

    def upload_image(number=False):
        try:
            if number is False:
                file_path = filedialog.askopenfilename(
                    initialdir=os.path.join(os.path.dirname(__file__), 'dataset'),
                    filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
                )
            else:
                file_path = os.path.join(os.path.dirname(__file__), 'dataset', f'emoji_{number}.jpg')
            
            if file_path:
                image_path.set(file_path)
                prompt_label.config(text=f"Scanning {os.path.basename(file_path)}")
                find_emoji_button.pack(side=tk.LEFT, padx=5)
                display_image(file_path)
                find_emoji_button.config(state=tk.NORMAL)
                reset_button.pack(side=tk.RIGHT)
                upload_button.pack_forget()
        except Exception as e:
            prompt_label.config(text=f"Error loading image: {str(e)}")

    def display_image(file_path):
        nonlocal displayed_image

        try:
            img = cv2.imread(file_path)
            if img is None:
                result_label.config(text="Failed to load image")
                return

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            displayed_image = ImageTk.PhotoImage(pil_img)
            for widget in image_frame.winfo_children():
                widget.destroy()
            image_label = tk.Label(image_frame, image=displayed_image)
            image_label.pack()
            result_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
            root.title(f"Emoji Finder - {os.path.basename(file_path)}")
            find_emoji_button.config(state=tk.NORMAL)
        except Exception as e:
            result_label.config(text=f"Error loading image: {str(e)}")

    def reset_app():
        image_path.set("")
        for widget in image_frame.winfo_children():
            widget.destroy()
        find_emoji_button.config(state=tk.DISABLED)
        upload_button.pack(side=tk.LEFT, padx=5)
        find_emoji_button.pack(side=tk.LEFT, padx=5)
        reset_button.pack_forget()
        image_frame.pack_forget()
        result_label.pack_forget()
        image_frame.pack(pady=10)
        prompt_label.config(text="Please upload an image (800x600 pixels)")

    def run_emoji_finder():
        move_red_square()

    def detect_circle(img, x, y):
        roi = img[y:y+50, x:x+50]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x_c, y_c), radius = cv2.minEnclosingCircle(contour)
            if 20 < radius < 30:
                return True
        return False

    def finish_line(img_copy, x, y, step_size):
        roi = img_copy[y:y+step_size, x:x+step_size]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred_roi,
            cv2.HOUGH_GRADIENT,
            dp=1.05,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=24, 
            maxRadius=26 
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (circle_x, circle_y, radius) in circles:
                center_x = x + circle_x
                center_y = y + circle_y
                emoji_size = 50
                half_size = emoji_size // 2               
                top_left_emoji = (center_x - half_size, center_y - half_size)
                bottom_right_emoji = (center_x + half_size, center_y + half_size)
                cv2.rectangle(img_copy, top_left_emoji, bottom_right_emoji, (0, 255, 0), 2)              
                prompt_label.config(text=f"Found the circle at {top_left_emoji}")          
                return img_copy, top_left_emoji
        return img_copy, None

    def move_red_square():
        nonlocal displayed_image
        img = cv2.imread(image_path.get())
        if img is None:
            return
        height, width = img.shape[:2]
        step_size = 25  # Reduced from 50
        found = False
        for y in range(0, height - 2*step_size + 1, 2*step_size // 2):
            for x in range(0, width - 2*step_size + 1, 2*step_size // 2):
                img_copy = img.copy()
                if detect_circle(img, x, y):
                    img_copy, emoji_coords = finish_line(img_copy, x, y, 2*step_size)
                    if emoji_coords:
                        found = True
                cv2.rectangle(img_copy, (x, y), (x + 2*step_size, y + 2*step_size), (0, 0, 255), 1)
                img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                displayed_image = ImageTk.PhotoImage(pil_img)
                for widget in image_frame.winfo_children():
                    widget.destroy()
                image_label = tk.Label(image_frame, image=displayed_image)
                image_label.pack()
                root.update()
                if found:
                    break
            if found:
                break

    def run_test():
        existing_tests = [f for f in os.listdir('.') if f.startswith('Test') and f.endswith('.csv')]
        test_numbers = [int(f.replace('Test', '').replace('.csv', '')) for f in existing_tests if f.replace('Test', '').replace('.csv', '').isdigit()]
        next_test_number = 1 if not test_numbers else max(test_numbers) + 1
        test_filename = f'Test{next_test_number}.csv'
        
        def test_single_image(number):
            upload_image(number)
            root.update()
            time.sleep(0.5)
            run_emoji_finder()
            root.update()
            time.sleep(0.5) 

        with open(test_filename, 'w') as f:
            for i in range(101):  # 0 to 100
                try:
                    test_single_image(i)
                    text = prompt_label.cget("text")
                    if "Found the circle at" in text:
                        coords = text.split("at (")[1].strip(")").split(", ")
                        f.write(f"{i};emoji_{i}.jpg;['happy'];{coords[0]};{coords[1]}\n")
                    else:
                        f.write(f"{i};emoji_{i}.jpg;['not_found'];0;0\n")
                except Exception as e:
                    f.write(f"{i};emoji_{i}.jpg;['error'];0;0\n")
        
        result_label.config(text=f"Test completed and saved as {test_filename}")


    # --- PROGRAM BONES ------ # 
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    image_frame = tk.Frame(root)
    image_frame.pack(pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    prompt_label = tk.Label(top_frame, text="Please upload an image (800x600 pixels)", font=("Arial", 14))
    prompt_label.pack(side=tk.LEFT)

    reset_button = tk.Button(top_frame, text="Reset", command=reset_app, width=10)
    reset_button.pack_forget()

    upload_button = tk.Button(button_frame, text="Upload Image", command=upload_image, width=15)
    upload_button.pack(side=tk.LEFT, padx=5)

    test_button = tk.Button(button_frame, text="Run Test", command=run_test, width=15)
    test_button.pack(side=tk.LEFT, padx=5)

    find_emoji_button = tk.Button(button_frame, text="Find Emoji", command=run_emoji_finder, state=tk.DISABLED, width=15)
    
    result_label = tk.Label(root, text="", font=("Arial", 10))
    def on_enter(event):
            if find_emoji_button['state'] == tk.NORMAL:
                run_emoji_finder()

    root.bind('<Return>', on_enter)

    root.mainloop()

if __name__ == "__main__":
    main()