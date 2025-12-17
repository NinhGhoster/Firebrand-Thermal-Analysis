import flirpy.io.seq
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback # Import traceback for detailed error printing

# --- Configuration ---
SEQ_FILE_PATH = "C:/Users/Dustwun/Downloads/Rec-0019.seq"
SHOW_FIRST_FRAME_IMAGE = True
# --- End Configuration ---

def process_seq_file(file_path):
    """
    Reads a FLIR SEQ file, extracts temperature data for each frame,
    and prints summary statistics. Optionally displays the first frame.
    Attempts to get dimensions from the first frame. Includes detailed error reporting.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Attempting to open SEQ file: {file_path}")

    seq_reader = None
    try:
        seq_reader = flirpy.io.seq.Seq(file_path)
        print("File opened successfully.")

        num_frames = -1 # Initialize
        try:
            num_frames = len(seq_reader)
            print(f"  Number of frames: {num_frames}")
        except TypeError:
            print("  Number of frames: Could not determine directly (will process frame by frame).")

        # Defer reading dimensions and bit depth until we have a frame
        print("  Dimensions/Bit Depth: Will attempt to read from first frame.")
        print("-" * 30)

        metadata_printed = False
        frame_count = 0
        # --- Iterate through frames ---
        print("Starting frame iteration...") # Add marker
        for i, frame in enumerate(seq_reader):
            frame_count += 1
            current_frame_num = i + 1
            total_frames_str = f"/{num_frames}" if num_frames > 0 else ""

            # --- Try processing frame within its own try block ---
            try:
                # --- Get Metadata from First Frame ---
                if not metadata_printed:
                    print(f"Processing Frame {current_frame_num}{total_frames_str} (Reading metadata)...") # Indicate metadata read attempt
                    height, width, bit_depth = None, None, 'Unknown'
                    if hasattr(frame, 'thermal') and frame.thermal is not None:
                        try:
                            height, width = frame.thermal.shape
                            # Try to infer bit depth if possible (often requires more specific info)
                        except AttributeError:
                             print("  Warning: Frame thermal data does not have shape attribute.")
                        except Exception as shape_err:
                             print(f"  Warning: Error getting shape from thermal data: {shape_err}")

                    elif hasattr(frame, 'counts') and frame.counts is not None:
                         try:
                            height, width = frame.counts.shape
                         except AttributeError:
                             print("  Warning: Frame counts data does not have shape attribute.")
                         except Exception as shape_err:
                             print(f"  Warning: Error getting shape from counts data: {shape_err}")
                    # Add other potential metadata sources if known from flirpy docs for Seq frame objects

                    print(f"  Detected Dimensions: {height}x{width}")
                    # Bit depth often needs specific metadata access, default to Unknown
                    print(f"  Bit depth: {bit_depth}")
                    print("-" * 30) # Separator after metadata
                    metadata_printed = True
                # --- End Metadata Reading ---

                # --- Process Temperature Data ---
                # Add print statement before accessing attributes
                # print(f"Accessing data for Frame {current_frame_num}...")

                if hasattr(frame, 'thermal') and frame.thermal is not None:
                     temperature_data_celsius = frame.thermal
                     avg_temp = np.mean(temperature_data_celsius)
                     min_temp = np.min(temperature_data_celsius)
                     max_temp = np.max(temperature_data_celsius)
                     median_temp = np.median(temperature_data_celsius)

                     # Only print stats if not the first frame (or adjust if needed)
                     if i > 0 or not SHOW_FIRST_FRAME_IMAGE: # Avoid double printing for frame 1 if showing image
                          print(f"Frame {current_frame_num}{total_frames_str}:")
                          print(f"  Avg Temp: {avg_temp:.2f}°C")
                          print(f"  Min Temp: {min_temp:.2f}°C")
                          print(f"  Max Temp: {max_temp:.2f}°C")
                          print(f"  Median Temp: {median_temp:.2f}°C")

                     if i == 0 and SHOW_FIRST_FRAME_IMAGE:
                        print(f"Displaying thermal image for Frame {current_frame_num}...")
                        plt.figure(figsize=(10, 8))
                        im = plt.imshow(temperature_data_celsius, cmap='inferno')
                        plt.colorbar(im, label='Temperature (°C)')
                        plt.title(f'Thermal Image - Frame {current_frame_num}')
                        plt.xlabel("Pixel X")
                        plt.ylabel("Pixel Y")
                        plt.show()

                elif hasattr(frame, 'counts') and frame.counts is not None:
                     print(f"Frame {current_frame_num}{total_frames_str}: Thermal data (calibrated temperature) not directly available.")
                     print("  Raw sensor counts are available (.counts attribute).")
                     # Optionally display raw counts image for first frame if needed
                     # if i == 0 and SHOW_FIRST_FRAME_IMAGE: ... plt.imshow(frame.counts) ...

                else:
                    print(f"Frame {current_frame_num}{total_frames_str}: Could not access thermal or counts data.")
                # --- End Temperature Processing ---

            # --- Catch errors specific to frame processing ---
            except Exception as frame_err:
                print(f"\n!!! Error processing frame {current_frame_num} !!!")
                print(f"Error details: {frame_err}")
                traceback.print_exc() # Print full traceback for frame error
                print("Attempting to continue with next frame...")
                # Continue to the next frame if one fails
                continue
        # --- End Frame Iteration ---

        print("-" * 30)
        print(f"Finished processing {frame_count} frames.")

    except ImportError:
        print("Error: Necessary libraries (flirpy, numpy, matplotlib) not found.")
        print("Please install them using: pip install flirpy numpy matplotlib")
    # --- Catch errors during initial setup or iteration start ---
    except Exception as e:
        print("\n!!! An unexpected error occurred during file setup or iteration start !!!")
        print(f"Error details: {e}")
        # --- Use traceback to get detailed error information ---
        print("\n----- Full Traceback -----")
        traceback.print_exc()
        print("--------------------------\n")

    # No finally block needed for closing

# --- Run the processing function ---
if __name__ == "__main__":
    process_seq_file(SEQ_FILE_PATH.replace("\\", "/"))