from PIL import Image
import io

class Helper:
    @staticmethod
    def get_img_from_bytes(file : bytes) -> Image:
        return Image.open(io.BytesIO(file)).convert('RGB')