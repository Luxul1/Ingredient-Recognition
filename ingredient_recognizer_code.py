from ultralytics import YOLO
import cv2

def recognize_ingredients(image_path):
    """
    This is the core Ingredient Recognizer logic.
    It loads the trained YOLOv8 AI model and scans an image to detect food ingredients.
    """
    print(f"Loading AI model...")
    # Load the trained YOLOv8 model (the "brain")
    model = YOLO('best.pt')
    
    print(f"Scanning image: {image_path}...")
    
    # Run the AI Inference
    # conf=0.25 ignores low-confidence guesses
    results = model.predict(source=image_path, imgsz=640, conf=0.25)
    
    # Parse the results
    result = results[0]
    ingredients_found = []
    
    for box in result.boxes:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        
        ingredients_found.append({
            "ingredient": class_name,
            "confidence": f"{conf * 100:.1f}%"
        })
        
    print(f"Found {len(ingredients_found)} ingredients:")
    for item in ingredients_found:
        print(f" - {item['ingredient']} (Confidence: {item['confidence']})")
        
    return ingredients_found

if __name__ == "__main__":
    # Example usage:
    # ingredients = recognize_ingredients("sample_food_photo.jpg")
    pass
