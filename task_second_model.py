import cv2
import sys
import numpy as np
from collections import deque
import time
import psutil
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - GPU detection limited")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("GPUtil not available - install with: pip install gputil")

class ObjectTracker:
    def __init__(self):
        (major, minor, _) = cv2.__version__.split(".")
        self.opencv_version = (int(major), int(minor))
        
        self.gpu_available = self._check_gpu_availability()
        
        self.OPENCV_OBJECT_TRACKERS = self._get_available_trackers()
        
        self.tracker_name = "gpu_template" if "gpu_template" in self.OPENCV_OBJECT_TRACKERS else "csrt" if "csrt" in self.OPENCV_OBJECT_TRACKERS else list(self.OPENCV_OBJECT_TRACKERS.keys())[0]
        self.tracker = None
        self.init_bb = None
        self.fps = 0
        
        self.use_gpu = False
        self.gpu_mat = None
        self.template = None
        self.template_gpu = None
        self.template_loc = None
        self.template_size = None
        self.gpu_stats = {"load": 0, "memory": 0, "temperature": 0}
        self.last_gpu_check = time.time()
        
        self.frame_times = deque(maxlen=30)
        self.cpu_percent = 0
        
    def _check_gpu_availability(self):
        print("\n" + "=" * 50)
        print("GPU AVAILABILITY CHECK")
        print("=" * 50)
        
        gpu_available = False
        
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                print(f"✓ OpenCV CUDA Support: {cuda_count} device(s)")
                print(f"  Device 0: {cv2.cuda.getDevice()}")
                gpu_available = True
            else:
                print("✗ OpenCV CUDA Support: Not available")
        except:
            print("✗ OpenCV CUDA Support: Not available")
        
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                print(f"✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
                print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                gpu_available = True
            else:
                print("✗ PyTorch CUDA: Not available")
        
        # if GPUTIL_AVAILABLE:
        #     try:
        #         gpus = GPUtil.getGPUs()
        #         if gpus:
        #             print(f"✓ GPUtil Detection: {len(gpus)} GPU(s) found")
        #             for gpu in gpus:
        #                 print(f"  GPU {gpu.id}: {gpu.name}")
        #                 print(f"    Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
        #                 print(f"    Load: {gpu.load * 100:.1f}%")
        #                 print(f"    Temperature: {gpu.temperature}°C")
        #             gpu_available = True
        #         else:
        #             print("✗ GPUtil: No GPUs detected")
        #     except Exception as e:
        #         print(f"✗ GPUtil Error: {e}")
        
        # print("=" * 50 + "\n")
        # return gpu_available
        
    def _get_available_trackers(self):
        trackers = {}
        
        if self.gpu_available:
            trackers["gpu_template"] = "GPU_Template_Matching"
        
        trackers["template"] = "Template_Matching"
        
        try:
            trackers["csrt"] = cv2.legacy.TrackerCSRT_create
            trackers["kcf"] = cv2.legacy.TrackerKCF_create
            trackers["mosse"] = cv2.legacy.TrackerMOSSE_create
            print("Using OpenCV legacy tracker API")
        except:
            pass
                
        try:
            if hasattr(cv2, 'TrackerMIL_create'):
                trackers["mil"] = cv2.TrackerMIL_create
            elif hasattr(cv2.legacy, 'TrackerMIL_create'):
                trackers["mil"] = cv2.legacy.TrackerMIL_create
        except:
            pass
            
        print(f"Available trackers: {list(trackers.keys())}")
        return trackers
    
    def _update_gpu_stats(self):
        current_time = time.time()
        if current_time - self.last_gpu_check < 0.5:
            return
            
        self.last_gpu_check = current_time
        
        self.cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_stats = {
                        "load": gpu.load * 100,
                        "memory": f"{gpu.memoryUsed}/{gpu.memoryTotal}MB",
                        "temperature": gpu.temperature
                    }
            except:
                pass
    
    def select_roi(self, frame):
        print("\nSelect the object to track and press ENTER or SPACE")
        print("Press 'c' or ESC to cancel selection")
        
        roi = cv2.selectROI("Select Object", frame, fromCenter=False,
                           showCrosshair=True)
        cv2.destroyWindow("Select Object")
        
        if roi[2] > 0 and roi[3] > 0:
            return roi
        return None
    
    def initialize_tracker(self, frame, roi):
        self.init_bb = roi
        
        if self.tracker_name == "gpu_template":
            self.use_gpu = True
            self._init_gpu_template_tracker(frame, roi)
        elif self.tracker_name == "template":
            self.use_gpu = False
            self._init_template_tracker(frame, roi)
        else:
            self.use_gpu = False
            self.tracker = self.OPENCV_OBJECT_TRACKERS[self.tracker_name]()
            self.tracker.init(frame, self.init_bb)
    
    def _init_gpu_template_tracker(self, frame, roi):
        x, y, w, h = [int(v) for v in roi]
        self.template = frame[y:y+h, x:x+w].copy()
        self.template_loc = (x, y)
        self.template_size = (w, h)
        
        try:
            self.template_gpu = cv2.cuda_GpuMat()
            self.template_gpu.upload(self.template)
            print("GPU Template tracker initialized successfully")
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            self.use_gpu = False
            self._init_template_tracker(frame, roi)
    
    def _init_template_tracker(self, frame, roi):
        x, y, w, h = [int(v) for v in roi]
        self.template = frame[y:y+h, x:x+w].copy()
        self.template_loc = (x, y)
        self.template_size = (w, h)
        print("CPU Template tracker initialized")
    
    def track_object(self, frame):
        timer = cv2.getTickCount()
        
        if self.tracker_name == "gpu_template" and self.use_gpu:
            success, bbox = self._track_gpu_template(frame)
        elif self.tracker_name in ["template", "gpu_template"]:
            success, bbox = self._track_template(frame)
        else:
            success, bbox = self.tracker.update(frame)
        
        self.fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        self.frame_times.append(1.0 / self.fps if self.fps > 0 else 0)
        
        self._update_gpu_stats()
        
        return success, bbox
    
    def _track_gpu_template(self, frame):
        try:
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)
            
            x, y = self.template_loc
            w, h = self.template_size
            
            margin = max(w, h) // 2
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            search_region = frame[y1:y2, x1:x2]
            search_gpu = cv2.cuda_GpuMat()
            search_gpu.upload(search_region)
            
            matcher = cv2.cuda.createTemplateMatching(cv2.TM_CCOEFF_NORMED)
            result_gpu = matcher.match(search_gpu, self.template_gpu)
            
            result = result_gpu.download()
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.7:
                self.template_loc = (x1 + max_loc[0], y1 + max_loc[1])
                return True, (self.template_loc[0], self.template_loc[1], w, h)
            
            return False, None
            
        except Exception as e:
            print(f"GPU tracking error: {e}, falling back to CPU")
            self.use_gpu = False
            return self._track_template(frame)
    
    def _track_template(self, frame):
        if self.template is None:
            return False, None
        
        x, y = self.template_loc
        w, h = self.template_size
        
        margin = max(w, h) // 2
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        search_window = frame[y1:y2, x1:x2]
        
        
        if search_window.shape[0] < h or search_window.shape[1] < w:
            return False, None
        
        result = cv2.matchTemplate(search_window, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.7:
            self.template_loc = (x1 + max_loc[0], y1 + max_loc[1])
            
            new_template = frame[self.template_loc[1]:self.template_loc[1]+h,
                                self.template_loc[0]:self.template_loc[0]+w]
            if new_template.shape == self.template.shape:
                self.template = cv2.addWeighted(self.template, 0.8, new_template, 0.2, 0)
                
            return True, (self.template_loc[0], self.template_loc[1], w, h)
        
        return False, None
    
    def draw_bounding_box(self, frame, bbox, success):
        overlay = frame.copy()
        
        panel_height = 200 if self.gpu_available else 120
        cv2.rectangle(overlay, (0, 0), (350, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
            
            corner_length = min(w, h) // 4
            thickness = 3
            cv2.line(frame, (x, y), (x + corner_length, y), (0, 255, 0), thickness)
            cv2.line(frame, (x, y), (x, y + corner_length), (0, 255, 0), thickness)
            cv2.line(frame, (x + w, y), (x + w - corner_length, y), (0, 255, 0), thickness)
            cv2.line(frame, (x + w, y), (x + w, y + corner_length), (0, 255, 0), thickness)
            cv2.line(frame, (x, y + h), (x + corner_length, y + h), (0, 255, 0), thickness)
            cv2.line(frame, (x, y + h), (x, y + h - corner_length), (0, 255, 0), thickness)
            cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), (0, 255, 0), thickness)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), (0, 255, 0), thickness)
            
            cv2.putText(frame, "TRACKING", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "TARGET LOST", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        tracker_text = self.tracker_name.upper().replace("_", " ")
        if self.use_gpu:
            tracker_text += " (GPU)"
        else:
            tracker_text += " (CPU)"
        cv2.putText(frame, tracker_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        
        cv2.putText(frame, "FPS: {:.2f}".format(self.fps), (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        
        cv2.putText(frame, f"CPU: {self.cpu_percent:.1f}%", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.gpu_available and GPUTIL_AVAILABLE:
            y_offset = 125
            cv2.putText(frame, f"GPU Load: {self.gpu_stats['load']:.1f}%", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            cv2.putText(frame, f"GPU Memory: {self.gpu_stats['memory']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            cv2.putText(frame, f"GPU Temp: {self.gpu_stats['temperature']}°C", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.use_gpu:
            cv2.circle(frame, (320, 25), 8, (0, 255, 0), -1)
            cv2.putText(frame, "GPU", (290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.circle(frame, (320, 25), 8, (128, 128, 128), -1)
            cv2.putText(frame, "CPU", (290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame
    
    def run(self):
        print("Starting webcam...")
        video = cv2.VideoCapture(0)
        
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video.set(cv2.CAP_PROP_FPS, 30)
        
        if not video.isOpened():
            print("Error: Could not open webcam")
            return
        
        ok, frame = video.read()
        if not ok:
            print("Error: Cannot read video stream")
            return
        
        roi = self.select_roi(frame)
        if roi is None:
            print("No ROI selected. Exiting...")
            return
        
        self.initialize_tracker(frame, roi)
        print(f"\n{self.tracker_name.upper()} tracker initialized.")
        
        print("press 'q': quit")
        
        show_info = True
        
        while True:
            ok, frame = video.read()
            if not ok:
                break
            
            success, bbox = self.track_object(frame)
            
            if show_info:
                frame = self.draw_bounding_box(frame, bbox, success)
            elif success:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow("Object Tracking", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
        
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"\nPerformance Summary:")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Average Frame Time: {avg_time*1000:.2f} ms")
            if self.use_gpu:
                print("GPU acceleration was used")
        
        video.release()
        cv2.destroyAllWindows()

def print_system_info():
    print("\nSystem Information:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if TORCH_AVAILABLE:
        print(f"PyTorch: {torch.__version__}")
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

def main():
    print("=" * 60)
    print("Real-Time Object Tracker with GPU Support")
    print("=" * 60)
    
    print_system_info()
    
    tracker = ObjectTracker()
    
    if tracker.gpu_available:
        print("\nWarming up GPU...")
        try:
            test_mat = cv2.cuda_GpuMat()
            test_mat.upload(np.zeros((100, 100, 3), dtype=np.uint8))
            print("GPU warm-up successful")
        except:
            print("GPU warm-up failed")
    
    tracker.run()

if __name__ == "__main__":
    main()