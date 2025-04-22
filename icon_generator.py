from PIL import Image, ImageDraw, ImageFilter
import math

SIZE = 512
PADDING = 64
BG_COLOR = (16, 163, 127)
LIGHT_COLOR = (255, 255, 255, 220)

def create_icon():
    img = Image.new('RGBA', (SIZE, SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    points = [
        (PADDING * 1.5, PADDING),
        (PADDING * 1.5 + 50, PADDING),
        (SIZE - PADDING * 1.5, SIZE - PADDING),
        (SIZE - PADDING * 1.5 - 50, SIZE - PADDING),
        (SIZE - PADDING * 1.5, PADDING),
        (SIZE - PADDING * 1.5 - 50, PADDING),
        (PADDING * 1.5, SIZE - PADDING),
        (PADDING * 1.5 + 50, SIZE - PADDING),
    ]
    draw.polygon(points[:4], fill=LIGHT_COLOR)
    draw.polygon(points[4:], fill=LIGHT_COLOR)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    gradient = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    gradient_draw = ImageDraw.Draw(gradient)
    for y in range(SIZE):
        alpha = int(25 * y / SIZE)
        gradient_draw.line([(0, y), (SIZE, y)], fill=(255, 255, 255, alpha))
    img = Image.alpha_composite(img, gradient)
    mask = Image.new('L', (SIZE, SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    radius = SIZE // 8
    mask_draw.rounded_rectangle([(0, 0), (SIZE, SIZE)], radius=radius, fill=255)
    img.putalpha(mask)
    img.save('icon.png')
    print("Icon created successfully as 'icon.png'")

if __name__ == "__main__":
    create_icon()