# test_florence2.py - Test Florence-2 for object identification
"""
Tests Microsoft's Florence-2 model for zero-shot object identification.
Florence-2 can do:
- Image captioning
- Object detection (with bounding boxes)
- Dense region captioning
- OCR
"""

import cv2
import torch
from PIL import Image
import numpy as np
import time

# ============================================================================
# SETUP
# ============================================================================

def setup_florence():
    """Download and setup Florence-2"""
    print("="*70)
    print("FLORENCE-2 TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    print("\nLoading Florence-2 (first run will download ~1.5GB)...")
    
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    # Florence-2-large is the best balance of speed/quality
    model_id = "microsoft/Florence-2-large"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"  # Avoid SDPA compatibility issue
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    print("✓ Florence-2 loaded!")
    
    return model, processor, device

# ============================================================================
# FLORENCE-2 TASKS
# ============================================================================

def run_florence(model, processor, image, task, device, text_input=None):
    """Run a Florence-2 task on an image"""
    
    # Task prompts for Florence-2
    task_prompts = {
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        "object_detection": "<OD>",
        "dense_region_caption": "<DENSE_REGION_CAPTION>",
        "region_proposal": "<REGION_PROPOSAL>",
        "caption_to_phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
        "ocr": "<OCR>",
        "ocr_with_region": "<OCR_WITH_REGION>",
    }
    
    prompt = task_prompts.get(task, task)
    
    if text_input:
        prompt = prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=1,
        do_sample=False,
        use_cache=False
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Parse the response
    parsed = processor.post_process_generation(
        generated_text, 
        task=prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed

# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_detections(frame, detections, color=(0, 255, 0)):
    """Draw bounding boxes and labels on frame"""
    if not detections:
        return frame
    
    frame = frame.copy()
    
    # Handle different detection formats
    if isinstance(detections, dict):
        if '<OD>' in detections:
            det = detections['<OD>']
            bboxes = det.get('bboxes', [])
            labels = det.get('labels', [])
        elif 'bboxes' in detections:
            bboxes = detections.get('bboxes', [])
            labels = detections.get('labels', [])
        else:
            return frame
    else:
        return frame
    
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

# ============================================================================
# MAIN TEST LOOP
# ============================================================================

def main():
    model, processor, device = setup_florence()
    
    print("\n" + "="*70)
    print("INTERACTIVE TEST")
    print("="*70)
    print("\nCommands:")
    print("  ENTER     - Capture and analyze (caption + detection)")
    print("  d         - Object detection only")
    print("  c         - Detailed caption only")
    print("  v         - Visual mode (continuous detection)")
    print("  q         - Quit")
    print("="*70)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    # Warm up
    print("\nWarming up camera...")
    for _ in range(10):
        cap.read()
    
    while True:
        user_input = input("\nCommand (ENTER/d/c/v/q): ").strip().lower()
        
        if user_input == 'q':
            break
        
        elif user_input == 'v':
            # Visual mode - continuous detection with display
            print("Visual mode - press 'q' in the window to exit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert to PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                
                # Run detection
                start = time.time()
                detections = run_florence(model, processor, pil_image, "object_detection", device)
                elapsed = (time.time() - start) * 1000
                
                # Draw detections
                display_frame = draw_detections(frame, detections)
                
                # Add FPS
                cv2.putText(display_frame, f"{elapsed:.0f}ms", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Florence-2 Detection", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
        
        else:
            # Single capture mode
            print("\n   Capturing...")
            ret, frame = cap.read()
            
            if not ret:
                print("   Failed to capture")
                continue
            
            # Convert to PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            if user_input == 'd':
                # Detection only
                print("   Running object detection...")
                start = time.time()
                result = run_florence(model, processor, pil_image, "object_detection", device)
                elapsed = (time.time() - start) * 1000
                
                print(f"\n   Object Detection ({elapsed:.0f}ms):")
                if '<OD>' in result:
                    for label, bbox in zip(result['<OD>']['labels'], result['<OD>']['bboxes']):
                        print(f"      • {label}")
                else:
                    print(f"      {result}")
                
            elif user_input == 'c':
                # Detailed caption only
                print("   Running detailed caption...")
                start = time.time()
                result = run_florence(model, processor, pil_image, "more_detailed_caption", device)
                elapsed = (time.time() - start) * 1000
                
                print(f"\n   Detailed Caption ({elapsed:.0f}ms):")
                if '<MORE_DETAILED_CAPTION>' in result:
                    print(f"      {result['<MORE_DETAILED_CAPTION>']}")
                else:
                    print(f"      {result}")
            
            else:
                # Default: caption + detection
                print("   Running caption...")
                start = time.time()
                caption = run_florence(model, processor, pil_image, "caption", device)
                caption_time = (time.time() - start) * 1000
                
                print("   Running object detection...")
                start = time.time()
                detection = run_florence(model, processor, pil_image, "object_detection", device)
                detection_time = (time.time() - start) * 1000
                
                print(f"\n   Caption ({caption_time:.0f}ms):")
                if '<CAPTION>' in caption:
                    print(f"      \"{caption['<CAPTION>']}\"")
                else:
                    print(f"      {caption}")
                
                print(f"\n   Objects Detected ({detection_time:.0f}ms):")
                if '<OD>' in detection:
                    for label in detection['<OD>']['labels']:
                        print(f"      • {label}")
                else:
                    print(f"      {detection}")
                
                # Show the frame with detections
                display_frame = draw_detections(frame, detection)
                cv2.imshow("Florence-2 Detection", display_frame)
                cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest complete.")

if __name__ == "__main__":
    main()