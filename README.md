# image_pross

**ex1_utils:**
1. **imReadAndConvert**(filename: str, representation: int)
   Reads image and returns the image converted as requested (GRAY_SCALE (1) or RGB (2))
   
2. **imDisplay**(filename: str, representation: int)
    Reads image as RGB or GRAY_SCALE with imReadAndConvert and displays it
    
3. **transformRGB2YIQ**(imgRGB: np.ndarray)
    Converts an RGB image to YIQ color space
    
    ![Alt text](images/1.png)
    
4. **transformYIQ2RGB**(imgYIQ: np.ndarray)
   Converts an YIQ image to RGB color space
   
5. **hsitogramEqualize**(imgOrig: np.ndarray)
    Equalizes the histogram of an imag
    
    ![Alt text](images/2.png)
    
6. **quantizeImage**(imOrig: np.ndarray, nQuant: int, nIter: int)
    Quantized an image in to nQuant colors, in nIter iterations
    
    ![Alt text](images/3.png)
    
    
**gamma:**    
function that performs gamma correction on an image with a given gamma

![Alt text](images/4.png)

Python 3.9.10


