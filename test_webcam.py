from ultralytics import YOLO

def main():
    print("Loading AI Brain (best.pt)...")
    # Load our trained YOLO model
    # Note: If the best.pt file is extracted from the zip, make sure it's in the same folder as this script.
    try:
        model = YOLO('best.pt')
    except Exception as e:
        print("❌ Error: Could not find 'best.pt'. Please make sure you extracted the ZIP file and put 'best.pt' in this folder!")
        return

    print("🚀 Opening webcam... (Press 'q' in the camera window to quit)")
    
    # source=0 opens the default laptop webcam.
    # show=True pops open the window so you can see the results live.
    # conf=0.35 means it will only draw a box if it's 35% confident.
    model.predict(source=0, show=True, conf=0.35)

if __name__ == "__main__":
    main()
