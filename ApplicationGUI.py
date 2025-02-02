import customtkinter as ctk
import cv2
import imutils
from PIL import Image
from PIL import ImageTk
import HPEstimation as hpe
from tkinter.filedialog import askopenfilename, asksaveasfilename
from itertools import combinations


class MainWindow:
    def __init__(self, parent):
        #----------------------------------------MAIN----------------------------------------------#
        ctk.set_appearance_mode("dark")
        self.cap = cv2.VideoCapture(0)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = 10
        self.save_path = "Data\\processedVideos\\output.avi"
        self.out = None
        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #self.cap_width = int(self.cap.set(cv2.CAP_PROP_FRAME_WIDTH),640)
        #self.cap_height = int(self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT),480)
        self.frame_size = (self.cap_width,self.cap_height)
        self.transform_sizes = [32, 64, 128, 192, 224, 256, 288, 320, 352]
        self.ratio_pairs = self.RatioPair(self.transform_sizes)
        self.ratio_pairs.update({1/key: value[::-1] for key, value in self.ratio_pairs.items()})
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

        #------------------------------------- CONTROL PANEL--------------------------------------#
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

        #---------------------------------------- WEBCAM PANEL -------------------------------------------------#
        # create instnace for webcam panel
        self.webcam_frame = ctk.CTkFrame(self.parent)
        self.webcam_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        #create instance for holding the images in the gui
        self.image_holder = ctk.CTkLabel(self.webcam_frame, text="", padx=10, pady=10)
        #self.webcam_holder.place(relx=0.5, rely=0.5, anchor="center")
        self.image_holder.grid(row=0, column=0, sticky = "nswe")

        #------------------------------------- RECORDING/PLAY PANEL-----------------------------------------------#
        
        self.recording_frame = ctk.CTkFrame(self.parent, width=self.webcam_frame.cget("width"),
                                            height = self.webcam_frame.cget("height")/10)
        self.recording_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.recording_frame.rowconfigure(0, weight=1)
        self.recording_frame.rowconfigure(1, weight=1)
        self.screenshot_button = ctk.CTkButton(self.recording_frame, text="Take Screenshot", command=self.ScreenshotEvent)
        self.screenshot_button.grid(row=0,column=0,padx=10, pady=5, sticky="nsew")
        self.recording_button = ctk.CTkButton(self.recording_frame, text="Start Recording", command=self.RecordingEvent, fg_color="green", hover_color="green")
        self.recording_button.grid(row=1,column=0,padx=10,pady=5,sticky="nsew")
        self.play_button = ctk.CTkButton(self.recording_frame, text="Play Video", command=self.PlayVidEvent, fg_color="green", hover_color="green")
        self.play_button.grid(row=0,column=1,padx=10,pady=5,sticky="nsew")

    def CamSwitchEvent(self):
        # Event for switching Webcam off or on
        if self.switch_cam_var.get() == "on":
            self.switch_video_var.set("off")
            self.cap = cv2.VideoCapture(0)
            self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
            self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.StartCapture()
        else:
            self.cap.release()

            if self.image_holder:
                self.image_holder.configure(image=None)
                self.image_holder.image = None
        return

    def VideoSwitchEvent(self):
        # Event for switching from webcam footage to video footage
        if self.switch_video_var.get() == "on":
            self.cap.release()
            self.switch_cam_var.set("off")

        if self.import_text.get() != "Upload Video to App":
            self.ChangeVideoCap(self.import_text.get())
            return
        
        self.image_holder.configure(image=None)
        self.image_holder.image = None
    
    def RatioPair(self, num_list):
        # Kombinationen von zwei Elementen aus der Liste erstellen
        pairs = list(combinations(num_list, 2))
        # Verhältnisse berechnen und in absteigender Reihenfolge sortieren
        ratios = [pair[0] / pair[1] for pair in pairs]
        ratio_dict = dict(zip(ratios,pairs))
    
        return ratio_dict


    def ActivateEstimationEvent(self):
        # activates and deactivates the Human Pose Estimation
        if self.activate_estimation_var.get() == "on":
            self.estimation = hpe.PoseEstimation()
        else:
            del self.estimation

    def RecordingEvent(self):
        #Event to start recording
        if self.recording_button.cget("text") == "Start Recording" and self.switch_cam_var.get()=="on":
            self.recording_button.configure(text="Stop Recording", fg_color = "red", hover_color = "red")
            #self.writeVideo(self.save_path, self.fourcc, self.fps, self.frame_size, self.frame)
        else:
            self.recording_button.configure(text="Start Recording", fg_color = "green", hover_color = "green")
            if self.out: 
                self.out.release()
                self.out = None
        pass

    def PlayVidEvent(self):
        # Event that starts the video 
        if self.play_button.cget("text") == "Play Video" and self.switch_video_var.get()=="on":
            self.play_button.configure(text="Stop Video", fg_color = "red", hover_color = "red")
            self.StartCapture()
            print("Starting Capture")
        else:
            self.play_button.configure(text="Play Video", fg_color = "green", hover_color = "green")
        

    def writeVideo(self,save_path, fourcc, fps, frame_size, frame):
        if not self.out:
            self.out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
        self.out.write(frame)
        
    def StartCapture(self):
        # method for starting the capturing of video or webcam
        # if self.switch_cam_var.get() == "off":
        #    return
        ratio = self.cap_width/self.cap_height
        self.transform_size = min(self.ratio_pairs.keys(), key = lambda x: abs(x-ratio)) #find nearest ratio dict entry
        self.transform_size = self.ratio_pairs[self.transform_size] # select matching size
        ret, self.frame = self.cap.read()
        if ret:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # reshaping image
            input_image = self.estimation.transform_frame(self.frame.copy(), self.transform_size[1], self.transform_size[0])   # frame,height,width

            # make keypoint detection
            results = self.estimation.movenet(input_image)
            keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

            # Render keypoints
            self.estimation.loop_through_people(self.frame, keypoints_with_scores, self.confidence_slider.get())
            self.raw_img = Image.fromarray(self.frame)
            self.raw_img = imutils.resize(self.raw_img, width=640)

            self.frame = imutils.resize(self.frame, width=640)

            if self.recording_button.cget("text") == "Stop Recording":
                self.writeVideo(self.save_path, self.fourcc, self.fps, (self.cap_width,self.cap_height), cv2.cvtColor(self.frame,cv2.COLOR_RGB2BGR))

            # place image into the holder
            self.image_holder.configure(image=self.estimation_img)
            self.image_holder.image = self.estimation_img
            print(self.frame.shape[1] + "," + self.frame.shape[0])
            #change image into ctkimage object 
            self.estimation_img = ctk.CTkImage(dark_image=self.raw_img, size=(self.frame.shape[1], self.frame.shape[0]))

        # Die ShowWebcam-Methode wird erneut nach 20 Millisekunden aufgerufen
        if self.switch_cam_var.get() == "on" or self.video_switch.get() == "on":
            self.webcam_frame.after(20, self.StartCapture)
        return

    def ScreenshotEvent(self):
        if self.estimation_img:
            self.screenshot_window = ScreenshotWindow(self, self.raw_img)

    def FileExplorerEvent(self, command, list_filetypes):
        # open up file explorer
        #"All Files","*.*"
        if command == "open":
            self.f_path = askopenfilename(initialdir="/",title="Select File",
                                        filetypes=list_filetypes, defaultextension=".*")
            
        else:
            self.f_path = asksaveasfilename(initialdir="/",title="Select File",
                                        filetypes=list_filetypes, defaultextension=".*")
            
        return self.f_path
    
    def DisplayVideoEvent(self):
        # change the videocapture from webcam to video
        self.cap = cv2.VideoCapture(self.import_text)
        self.cap_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.cap_widht = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def ImportVideoEvent(self):
        # define filetypes, call file explorer and set text to video path
        filetypes = [("MP4 Videoformat .mp4", "*.mp4"),("AVI Videoformat .avi", "*.avi")]
        self.import_text.set(self.FileExplorerEvent("open", filetypes)) 

        if self.switch_video_var.get() == "on":
            self.ChangeVideoCap(self.import_text.get())

    def ChangeVideoCap(self,file_path): 
            # method for changing the videocapture and the height and frame 
            self.cap = cv2.VideoCapture(file_path) 
            self.cap_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.cap_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame = self.cap.read()[1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_img = Image.fromarray(frame)
            img= ctk.CTkImage(dark_image=raw_img, size=(frame.shape[1], frame.shape[0]))
            self.image_holder.configure(image=img)
            self.image_holder.image = img
        
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
        self.img = screenshot_img

        # Frame for Screenshot
        self.Screen_Frame = ctk.CTkFrame(self)
        self.Screen_Frame.grid(row=0,column=0,pady=5, padx=5, sticky="nsew")

        # Label Holder for Screenshot
        self.ctkScreenshot_img = ctk.CTkImage(dark_image=self.img, size=(640, 480))
        self.screenshot = ctk.CTkLabel(self.Screen_Frame, image = self.ctkScreenshot_img, text ="")
        self.screenshot.place(rely=0.5, relx=0.5, anchor = "center")

        # Save button
        self.config_frame = ctk.CTkFrame(self, height=10)
        self.config_frame.grid(row=1, column=0,sticky="nsew")
        self.save_button = ctk.CTkButton(self.config_frame, text="Save Image",command=self.saveScreenshot)
        self.save_button.place(rely=0.5, relx=0.5,anchor="center")

    def saveScreenshot(self):
        filetypes = [("Jpeg Image .jpg","*.jpg"),("PNG Image .png","*.png")]
        save_path = self.controller.FileExplorerEvent("save", filetypes)
        self.img.save(save_path)

        
def main():
    # start GUI
    root = ctk.CTk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()