# SOD: salient object detection
## Reference:
   https://github.com/NathanUA/U-2-Net
   https://github.com/cyrildiagne/basnet-http

## How to use this project:
- Download model file u2net.pth or u2netp.pth and put them in according folder saved_models/u2net
   and saved_models/u2netp
- Copy test image to folder test/input_images
- Run the following command:
   ```python 
      python3.7 main.py -m u2netp -im {image_name}
      e.g: python3.7 main.py -m u2netp -im batman_4.jpg
   ```
  Output will be saved in folder test/output_images