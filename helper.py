from PIL import Image
import io

class Helper:
    @staticmethod
    def get_img_from_bytes(file : bytes) -> Image:
        return Image.open(io.BytesIO(file)).convert('RGB')
    
    @staticmethod
    def get_bytes_from_image(img:Image) -> bytes:
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=85)
        output.seek(0)
        return output