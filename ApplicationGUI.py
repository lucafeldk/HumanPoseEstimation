import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import Image
import HPEstimation as hpe
from tkinter.filedialog import askopenfilename,asksaveasfilename


class MainWindow:
    def __init__(self, parent):
        ctk.set_appearance_mode("dark")
        self.cap = cv2.VideoCapture(0)
        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.estimation = hpe.PoseEstimation()
        self.estimation_img = None
        self.parent = parent

        #Main Window Layout Management
        self.parent.title("Human Pose Estimator")
        self.parent.geometry("950x650")
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_columnconfigure(1, weight =20)
        self.parent.grid_rowconfigure(0, weight=10)
        self.parent.grid_rowconfigure(1, weight=1)

        #### CONTROL PANEL #####
        # create instance for control panel
        self.control_frame = ctk.CTkFrame(self.parent)
        self.control_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.control_frame.grid_columnconfigure(0,weight=1)
        self.control_frame.grid_columnconfigure(1,weight=8)

        # create off/on switch for webcam or video off/on
        self.switch_cam_var = ctk.StringVar(value="off")
        self.switch_video_var = ctk.StringVar(value="off")
        self.choose_Label = ctk.CTkLabel(self.control_frame, text="Choose Medium")
        self.choose_Label.grid(row=0, column=0,columnspan=2, sticky="nsew")
        self.cam_switch = ctk.CTkSwitch(self.control_frame, text="Use Webcam", command=self.CamSwitchEvent,
                                    variable=self.switch_cam_var, onvalue="on", offvalue="off")
        self.cam_switch.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.video_switch = ctk.CTkSwitch(self.control_frame, text="Use Video", command=self.VideoSwitchEvent,
                                    variable=self.switch_video_var, onvalue="on", offvalue="off")
        self.video_switch.grid(row=2, column=0,columnspan=2, padx=10, pady=10, sticky="nsew")

        # import Video widgets
        self.import_Label = ctk.CTkLabel(self.control_frame, text="Upload Video")
        self.import_Label.grid(row=5, column=0, columnspan=2, sticky="nsew")
        self.import_button = ctk.CTkButton(self.control_frame, text="...", command=self.ImportVideoEvent,width=10)
        self.import_button.grid(row=6, column=0,columnspan=1, padx=2, pady=10, sticky="nsew")
        self.import_text = ctk.StringVar(value="Upload Video to App")
        self.import_entry = ctk.CTkEntry(self.control_frame, textvariable=self.import_text)
        self.import_entry.grid(row=6, column=1,columnspan=1, padx=2, pady=10, sticky="nsew")

        # Estimation Configuration widgets
        self.confidence_slider = ctk.CTkSlider(self.control_frame, from_=1, to=0.0)
        self.confidence_slider.set(0.2)
        self.confidence_slider.grid(row=10, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.confidence_label = ctk.CTkLabel(self.control_frame, text="Estimation Sensitivity")
        self.confidence_label.grid(row=9, column=0,columnspan=2, sticky="nsew")

        #### WEBCAM PANEL #####
        # create instnace for webcam panel
        self.webcam_frame = ctk.CTkFrame(self.parent)
        self.webcam_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.webcam_holder = None

        #### RECORDING PANEL ####
        self.recording_frame = ctk.CTkFrame(self.parent, width=self.webcam_frame.cget("width"),
                                            height = self.webcam_frame.cget("height")/10)
        self.recording_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.screenshot_button = ctk.CTkButton(self.recording_frame, text="Screenshot", command=self.ScreenshotEvent)
        self.screenshot_button.grid(row=0,column=0,padx=10, pady=10, sticky="nsew")

    def CamSwitchEvent(self):
        # Event for switching Webcam of Laptop off or on
        if self.switch_cam_var.get() == "on":
            self.switch_video_var.set("off")
            self.cap = cv2.VideoCapture(0)
            self.ShowWebcam()
        else:
            self.cap.release()

            if self.webcam_holder:
                self.webcam_holder.configure(image=None)
                self.webcam_holder.image = None
                self.webcam_holder = None
        return

    def VideoSwitchEvent(self):
        # Event for switching from webcam footage to video footage
        if self.switch_video_var.get() == "on":
            self.switch_cam_var.set("off")

    def ActivateEstimationEvent(self):
        # activates and deactivates the Human Pose Estimation
        if self.activate_estimation_var.get() == "on":
            self.estimation = hpe.PoseEstimation()
        else:
            del self.estimation


    def ShowWebcam(self):
        # method for showing webcam footage 
        if self.switch_cam_var.get() == "off":
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # reshaping image
            input_image = self.estimation.transform_frame(frame.copy(), 192, 256)

            # make keypoint detection
            results = self.estimation.movenet(input_image)
            keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

            # Render keypoints
            self.estimation.loop_through_people(frame, keypoints_with_scores, self.confidence_slider.get())
            img = Image.fromarray(frame)

            if self.webcam_holder:
                self.webcam_holder.configure(image=self.estimation_img)
                self.webcam_holder.image = self.estimation_img
                self.estimation_img = ctk.CTkImage(dark_image=img, size=(640, 480))

            else:
                self.webcam_holder = ctk.CTkLabel(self.webcam_frame, image=self.estimation_img, text="", padx=10, pady=10)
                self.webcam_holder.place(relx=0.5, rely=0.5, anchor="center")

        # Die ShowWebcam-Methode wird erneut nach 20 Millisekunden aufgerufen
        if self.switch_cam_var.get() == "on":
            self.webcam_frame.after(20, self.ShowWebcam)
        return

    def ScreenshotEvent(self):
        if self.estimation_img:
            self.screenshot_window = ScreenshotWindow(self, self.estimation_img)


            
            

    def FileExplorerEvent(self, command):
        # open up file explorer
        if command == "open":
            self.f_path = askopenfilename(initialdir="/",title="Select File",
                                        filetypes=(("Text files","*.txt*"),("All Files","*.*")))
            
        else:
            self.f_path = asksaveasfilename(initialdir="/",title="Select File",
                                        filetypes=(("Text files","*.txt*"),("All Files","*.*")))
    
        return self.f_path
    
    def ImportVideoEvent(self):
        self.import_text.set(self.FileExplorerEvent("open")) 
        
    def __del__(self):
        self.cap.release()

class ScreenshotWindow(ctk.CTkToplevel):
    def __init__(self, controller, screenshot_img):
        super().__init__()
        self.controller = controller
        self.title("Screenshot Window")
        self.geometry("720x560")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=10)
        self.grid_rowconfigure(1, weight=1)
        self.focus()

        # Frame for Screenshot
        self.Screen_Frame = ctk.CTkFragitme(self)
        self.Screen_Frame.grid(row=0,column=0,pady=5, padx=5, sticky="nsew")

        # Label Holder for Screenshot
        self.screenshot = ctk.CTkLabel(self.Screen_Frame, image = screenshot_img, text ="")
        self.screenshot.place(rely=0.5, relx=0.5, anchor = "center")

        # Save button
        self.config_frame = ctk.CTkFrame(self, height=10)
        self.config_frame.grid(row=1, column=0,sticky="nsew")
        self.save_button = ctk.CTkButton(self.config_frame, text="Save Image",command=self.saveScreenshot)
        self.save_button.place(rely=0.5, relx=0.5,anchor="center")

    def saveScreenshot(self):
        save_path = self.controller.FileExplorerEvent("save")
        print(save_path)
          
def main():
    # start GUI
    root = ctk.CTk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()