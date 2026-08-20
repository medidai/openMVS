import sys
import argparse
from pathlib import Path
from PIL import Image
import piexif

# -------------------------------------------------------------------
# --- How to Use This Script ---
#
# 1. Install Dependencies (if you haven't already):
#    python3 -m pip install pillow piexif
#
# 2. Run from your terminal.
#
#    # Example 1: Set ONLY 35mm equivalent
#    python3 focal2exif.py "C:\Path\To\Images" --f_pixels 3200.0
#
#    # Example 2: Set BOTH focal lengths
#    python3 focal2exif.py "C:\Path\To\Images" --f_pixels 3200.0 --sensor_width 23.5
#
#    # Example 3: Set all information
#    python3 focal2exif.py "C:\Path\To\Images" --f_pixels 3200.0 --sensor_width 23.5 --make "Sony" --model "ILCE-7M4"
#
#    # Example 4: Set 35mm equivalent and camera model
#    python3 focal2exif.py "C:\Path\To\Images" --f_pixels 3200.0 --model "MyCustomCam"
#
# -------------------------------------------------------------------


# --- Helper Function ---

def float_to_rational(f, precision=10000):
    """
    Converts a float to a rational (numerator, denominator)
    for EXIF representation.
    """
    numerator = int(f * precision)
    denominator = precision
    return (numerator, denominator)

# --- Main Processing Function ---

def process_image(image_path_str, f_px, sensor_w_mm=None, camera_make=None, camera_model=None):
    """
    Calculates focal lengths and writes them (and optionally
    make/model) to the image's EXIF data.
    """
    try:
        image_path = Path(image_path_str)
        
        # 1. Get image width in pixels using Pillow
        with Image.open(image_path) as img:
            image_width_px = img.width

        if image_width_px == 0:
            print(f"SKIPPING: {image_path.name} (Image width is 0)")
            return

        # 2. Perform the calculations
        
        # Formula for 35mm equivalent focal length
        f_35mm_equiv = f_px * (36.0 / image_width_px)

        f_mm = None
        # Formula for focal length (mm)
        if sensor_w_mm:
            f_mm = f_px * (sensor_w_mm / image_width_px)

        # 3. Load existing EXIF data or create a new dict
        try:
            exif_dict = piexif.load(str(image_path))
        except piexif.InvalidExif:
            print(f"INFO: No valid EXIF data in {image_path.name}. Creating new.")
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        except Exception:
            print(f"INFO: No EXIF data in {image_path.name}. Creating new.")
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # 4. Set the new EXIF tags
        
        # --- 0th IFD (ImageIFD) Tags ---
        # Make and Model go here
        if "0th" not in exif_dict:
            exif_dict["0th"] = {}

        if camera_make:
            # piexif.ImageIFD.Make (Tag 271)
            exif_dict["0th"][piexif.ImageIFD.Make] = camera_make
        if camera_model:
            # piexif.ImageIFD.Model (Tag 272)
            exif_dict["0th"][piexif.ImageIFD.Model] = camera_model

        # --- Exif IFD Tags ---
        # Focal lengths go here
        if "Exif" not in exif_dict:
            exif_dict["Exif"] = {}

        # piexif.ExifIFD.FocalLengthIn35mmFilm (Tag 41989)
        exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(f_35mm_equiv))
        
        if f_mm is not None:
            # piexif.ExifIFD.FocalLength (Tag 37386)
            exif_dict["Exif"][piexif.ExifIFD.FocalLength] = float_to_rational(f_mm)

        # 5. Dump the EXIF data to bytes
        exif_bytes = piexif.dump(exif_dict)

        # 6. Insert the new EXIF bytes into the image file
        piexif.insert(exif_bytes, str(image_path))

        # 7. Print summary
        print(f"PROCESSED: {image_path.name}")
        if camera_make:
            print(f"  > Set Make: {camera_make}")
        if camera_model:
            print(f"  > Set Model: {camera_model}")
        if f_mm is not None:
            print(f"  > Set FocalLength: {f_mm:.2f}mm")
        print(f"  > Set FocalLengthIn35mmFilm: {int(round(f_35mm_equiv))}mm")

    except Exception as e:
        print(f"ERROR processing {image_path_str}: {e}")

# --- Main execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Batch add focal length (mm and 35mm-equivalent) and camera "
                    "make/model to EXIF data from a given focal length in pixels.",
        epilog="Usage Examples:\n"
               "  # Set only 35mm equivalent\n"
               "  python3 %(prog)s \"./my_photos\" --f_pixels 3200.0\n\n"
               "  # Set both focal lengths\n"
               "  python3 %(prog)s \"./my_photos\" --f_pixels 3200.0 --sensor_width 23.5\n\n"
               "  # Set all available info\n"
               "  python3 %(prog)s \"./my_photos\" --f_pixels 3200.0 --sensor_width 23.5 --make \"MyMake\" --model \"MyModel\"\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", help="Path to the folder containing images.")
    parser.add_argument(
        "--f_pixels", 
        type=float, 
        required=True, 
        help="The focal length in pixels (e.g., 'fx' from a camera matrix)."
    )
    parser.add_argument(
        "--sensor_width", 
        type=float, 
        required=False,
        default=None,
        help="[Optional] The physical width of the camera sensor in mm "
             "(e.g., 36.0 for full-frame). If provided, the standard "
             "FocalLength (mm) will also be set."
    )
    parser.add_argument(
        "--make", 
        type=str, 
        required=False,
        default=None,
        help="[Optional] The camera manufacturer (e.g., 'Sony', 'Canon', 'Apple')."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=False,
        default=None,
        help="[Optional] The camera model name (e.g., 'ILCE-7M4', 'iPhone 15 Pro')."
    )
    
    args = parser.parse_args()

    # --- Validate folder ---
    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        print(f"Error: '{args.folder}' is not a valid directory.")
        sys.exit(1)

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.tif', '.tiff'}

    print(f"Scanning '{folder_path}'...")
    print(f"Using: Focal Length (pixels) = {args.f_pixels}")
    if args.sensor_width:
        print(f"Using: Sensor Width = {args.sensor_width}mm (for standard FocalLength)")
    else:
        print("INFO: No sensor width provided. Only setting 35mm equivalent.")
    
    if args.make:
        print(f"Using: Make = {args.make}")
    if args.model:
        print(f"Using: Model = {args.model}")
        
    print("-" * 30)

    # Use rglob to recursively find all files
    for file_path in folder_path.rglob('*'):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            process_image(
                str(file_path), 
                args.f_pixels, 
                args.sensor_width,
                args.make,
                args.model
            )

    print("-" * 30)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()
