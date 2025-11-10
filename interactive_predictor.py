"""
Interactive Ball Position Predictor
- GUI file picker
- Click to select 2 faces
- Get ball prediction
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from l2cs import Pipeline
from tkinter import Tk, filedialog
import math


class InteractiveBallPredictor:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading L2CS model on {device}...")
        
        self.pipeline = Pipeline(
            weights=Path('L2CSNet_gaze360.pkl'),
            arch='ResNet50',
            device=device
        )
        
        print("✓ Model loaded\n")
        
        self.selected_faces = []
        self.image = None
        self.display = None
        self.results = None
        
    def get_gaze_ray(self, face_idx, length_multiplier=5):
        """Get gaze ray using eye position as origin"""
        bbox = self.results.bboxes[face_idx]
        landmarks = self.results.landmarks[face_idx]
        
        # Landmarks from RetinaFace: [right_eye, left_eye, nose, right_mouth, left_mouth]
        # Use midpoint between eyes as gaze origin
        if len(landmarks) >= 2:
            right_eye = landmarks[0]  # [x, y]
            left_eye = landmarks[1]   # [x, y]
            eye_center = (right_eye + left_eye) / 2
            origin = eye_center
        else:
            # Fallback to bbox center
            origin = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
        
        bbox_width = bbox[2] - bbox[0]
        pitch_rad = self.results.pitch[face_idx]
        yaw_rad = self.results.yaw[face_idx]
        
        # L2CS draw_gaze formula
        length = bbox_width * length_multiplier
        dx = -length * np.sin(pitch_rad) * np.cos(yaw_rad)
        dy = -length * np.sin(yaw_rad)
        
        endpoint = origin + np.array([dx, dy])
        direction = endpoint - origin
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        return {'origin': origin, 'direction': direction, 'endpoint': endpoint}
    
    def intersect_rays(self, ray1, ray2):
        """Find intersection or closest approach"""
        p1, d1 = ray1['origin'], ray1['direction']
        p2, d2 = ray2['origin'], ray2['direction']
        
        # Vector between origins
        w = p1 - p2
        
        # Dot products
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, w)
        e = np.dot(d2, w)
        
        denom = a*c - b*b
        
        if abs(denom) < 1e-10:
            return None, 0.0
        
        # Parameters for closest points
        t = (b*e - c*d) / denom
        s = (a*e - b*d) / denom
        
        if t < 0 or s < 0:
            return None, 0.0
        
        # Closest points
        closest1 = p1 + t * d1
        closest2 = p2 + s * d2
        intersection = (closest1 + closest2) / 2
        
        # Distance and angle quality
        distance = np.linalg.norm(closest1 - closest2)
        cos_angle = abs(np.dot(d1, d2))
        angle_quality = 1 - cos_angle
        distance_conf = np.exp(-distance / 300)
        
        confidence = angle_quality * distance_conf
        
        return intersection, confidence
    
    def draw_display(self):
        """Draw image with numbered faces and selections"""
        self.display = self.image.copy()
        
        if self.results is None:
            return
        
        # Draw all faces with numbers
        num_faces = len(self.results.pitch)

        for i in range(num_faces):
            bbox = self.results.bboxes[i]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Color: yellow if selected, green otherwise
            color = (0, 255, 255) if i in self.selected_faces else (0, 255, 0)
            thickness = 4 if i in self.selected_faces else 2
            
            cv2.rectangle(self.display, (x1, y1), (x2, y2), color, thickness)
            
            # Draw number
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Show eye landmarks if available
            if len(self.results.landmarks) > i:
                landmarks = self.results.landmarks[i]
                if len(landmarks) >= 2:
                    # Draw eyes
                    right_eye = landmarks[0].astype(int)
                    left_eye = landmarks[1].astype(int)
                    cv2.circle(self.display, tuple(right_eye), 3, (255, 0, 255), -1)
                    cv2.circle(self.display, tuple(left_eye), 3, (255, 0, 255), -1)
                    
                    # Draw line between eyes
                    cv2.line(self.display, tuple(right_eye), tuple(left_eye), (255, 0, 255), 1)
            
            # Background for number
            cv2.circle(self.display, (center_x, center_y), 25, (0, 0, 0), -1)
            cv2.circle(self.display, (center_x, center_y), 23, color, -1)
            
            # Number text
            cv2.putText(self.display, str(i+1), (center_x-12, center_y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            
            # Show gaze info
            pitch = self.results.pitch[i] * 180 / np.pi
            yaw = self.results.yaw[i] * 180 / np.pi
            score = self.results.scores[i]
            
            info = f"#{i+1}: P={pitch:.0f} Y={yaw:.0f} S={score:.2f}"
            cv2.putText(self.display, info, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw gaze rays if 2 faces selected
        if len(self.selected_faces) == 2:
            rays = []
            for idx in self.selected_faces:
                ray = self.get_gaze_ray(idx)
                rays.append(ray)
                
                # Draw arrow
                origin = ray['origin'].astype(int)
                endpoint = ray['endpoint'].astype(int)
                cv2.arrowedLine(self.display, tuple(origin), tuple(endpoint),
                               (0, 0, 255), 3, tipLength=0.1)
            
            # Compute intersection
            ball_pos, conf = self.intersect_rays(rays[0], rays[1])
            
            if ball_pos is not None:
                # Draw predicted position
                cv2.circle(self.display, (int(ball_pos[0]), int(ball_pos[1])),
                          20, (0, 255, 0), 3)
                cv2.circle(self.display, (int(ball_pos[0]), int(ball_pos[1])),
                          5, (0, 255, 0), -1)
                
                # Show prediction info
                pred_text = f"Ball: ({int(ball_pos[0])}, {int(ball_pos[1])}) Conf: {conf:.2f}"
                cv2.putText(self.display, pred_text, (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Instructions
        instructions = [
            "Click faces OR type numbers (space-separated)",
            "Press 'r' to reset | 's' to save | 'q' to quit",
            f"Selected: {self.selected_faces if self.selected_faces else 'None'}"
        ]
        
        y_pos = self.display.shape[0] - 80
        for i, text in enumerate(instructions):
            cv2.putText(self.display, text, (10, y_pos + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        if self.results is None:
            return

        # Find which face was clicked
        num_faces = len(self.results.pitch)
        for i in range(num_faces):
            bbox = self.results.bboxes[i]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Toggle selection
                if i in self.selected_faces:
                    self.selected_faces.remove(i)
                else:
                    if len(self.selected_faces) < 2:
                        self.selected_faces.append(i)
                    else:
                        # Replace first selection
                        self.selected_faces[0] = self.selected_faces[1]
                        self.selected_faces[1] = i
                
                self.draw_display()
                cv2.imshow('Ball Predictor', self.display)
                break
    
    def run(self):
        """Main interactive loop"""
        # File picker
        print("Opening file picker...")
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        image_path = filedialog.askopenfilename(
            title='Select football image',
            filetypes=[('Images', '*.jpg *.jpeg *.png'), ('All files', '*.*')]
        )
        
        root.destroy()
        
        if not image_path:
            print("No file selected")
            return
        
        print(f"\nProcessing: {Path(image_path).name}")
        
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            print("Failed to load image")
            return
        
        # Detect faces
        print("Detecting faces...")
        self.results = self.pipeline.step(self.image)
        
        num_faces = len(self.results.pitch)

        print(f"Found {num_faces} faces\n")

        if num_faces == 0:
            print("No faces detected")
            return

        # Show face info
        for i in range(num_faces):
            pitch = self.results.pitch[i] * 180 / np.pi
            yaw = self.results.yaw[i] * 180 / np.pi
            score = self.results.scores[i]
            print(f"Face {i+1}: Pitch={pitch:6.1f}° Yaw={yaw:6.1f}° Score={score:.2f}")
        
        print("\n" + "="*60)
        print("INSTRUCTIONS:")
        print("  - Click on 2 faces to select them")
        print("  - OR type numbers (e.g., '1 3' for faces 1 and 3)")
        print("  - Press 'r' to reset selection")
        print("  - Press 's' to save visualization")
        print("  - Press 'q' to quit")
        print("="*60 + "\n")
        
        # Resize for display if image is large
        display_image = self.image.copy()
        max_height = 1000
        if display_image.shape[0] > max_height:
            scale = max_height / display_image.shape[0]
            new_width = int(display_image.shape[1] * scale)
            display_image = cv2.resize(display_image, (new_width, max_height))
            # Scale results too
            # (for now, work with original size)
        
        self.draw_display()
        
        # Create window
        cv2.namedWindow('Ball Predictor', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Ball Predictor', self.mouse_callback)
        cv2.imshow('Ball Predictor', self.display)
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('r'):
                self.selected_faces = []
                self.draw_display()
                cv2.imshow('Ball Predictor', self.display)
                print("Selection reset")
            
            elif key == ord('s'):
                output_path = str(image_path).replace('.jpg', '_prediction.jpg')
                cv2.imwrite(output_path, self.display)
                print(f"✓ Saved to: {output_path}")
            
            elif key >= ord('0') and key <= ord('9'):
                # Number input mode - wait for full input
                print("\nEnter face numbers (space-separated, e.g. '1 3'): ", end='', flush=True)
                cv2.destroyWindow('Ball Predictor')
                
                try:
                    user_input = input()
                    numbers = [int(x.strip())-1 for x in user_input.split()]

                    if len(numbers) == 2:
                        if all(0 <= n < num_faces for n in numbers):
                            self.selected_faces = numbers
                            print(f"Selected faces: {[n+1 for n in numbers]}")
                        else:
                            print("Invalid face numbers")
                    else:
                        print("Please enter exactly 2 numbers")
                except:
                    print("Invalid input")
                
                self.draw_display()
                cv2.namedWindow('Ball Predictor', cv2.WINDOW_NORMAL)
                cv2.setMouseCallback('Ball Predictor', self.mouse_callback)
                cv2.imshow('Ball Predictor', self.display)
        
        cv2.destroyAllWindows()
        
        # Final prediction
        if len(self.selected_faces) == 2:
            rays = []
            for idx in self.selected_faces:
                ray = self.get_gaze_ray(idx)
                rays.append(ray)
            
            ball_pos, conf = self.intersect_rays(rays[0], rays[1])
            
            if ball_pos is not None:
                print(f"\n{'='*60}")
                print(f"PREDICTION:")
                print(f"  Ball position: ({int(ball_pos[0])}, {int(ball_pos[1])})")
                print(f"  Confidence: {conf:.2f}")
                
                # Check GT if available
                basename = Path(image_path).stem
                parts = basename.split('_')
                if len(parts) >= 3:
                    try:
                        gt = np.array([int(parts[1]), int(parts[2])])
                        error = np.linalg.norm(ball_pos - gt)
                        print(f"  Ground Truth: ({gt[0]}, {gt[1]})")
                        print(f"  Error: {error:.0f}px")
                    except:
                        pass
                print(f"{'='*60}")


if __name__ == '__main__':
    predictor = InteractiveBallPredictor()
    predictor.run()
