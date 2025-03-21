from PIL import Image, ImageDraw, ImageFilter
import math

# Constants for the icon
SIZE = 512
PADDING = 64
BG_COLOR = (16, 163, 127)  # OpenAI green (#10A37F)
LIGHT_COLOR = (255, 255, 255, 220)

def create_icon():
    # Create a base image with a green background
    img = Image.new('RGBA', (SIZE, SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Draw a stylized "N" for NeoCortex
    # Create points for the N shape
    points = [
        (PADDING * 1.5, PADDING),  # Top left
        (PADDING * 1.5 + 50, PADDING),  # Top left shoulder
        (SIZE - PADDING * 1.5, SIZE - PADDING),  # Bottom right
        (SIZE - PADDING * 1.5 - 50, SIZE - PADDING),  # Bottom right shoulder
        (SIZE - PADDING * 1.5, PADDING),  # Top right
        (SIZE - PADDING * 1.5 - 50, PADDING),  # Top right shoulder
        (PADDING * 1.5, SIZE - PADDING),  # Bottom left
        (PADDING * 1.5 + 50, SIZE - PADDING),  # Bottom left shoulder
    ]
    
    # Draw the N
    draw.polygon(points[:4], fill=LIGHT_COLOR)  # First diagonal
    draw.polygon(points[4:], fill=LIGHT_COLOR)  # Second diagonal
    
    # Apply a subtle blur for a modern look
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Add a subtle gradient overlay
    gradient = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    gradient_draw = ImageDraw.Draw(gradient)
    
    for y in range(SIZE):
        alpha = int(25 * y / SIZE)
        gradient_draw.line([(0, y), (SIZE, y)], fill=(255, 255, 255, alpha))
    
    img = Image.alpha_composite(img, gradient)
    
    # Apply rounded corners
    mask = Image.new('L', (SIZE, SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    radius = SIZE // 8
    mask_draw.rounded_rectangle([(0, 0), (SIZE, SIZE)], radius=radius, fill=255)
    
    # Apply the mask
    img.putalpha(mask)
    
    # Save the icon
    img.save('icon.png')
    print("Icon created successfully as 'icon.png'")

if __name__ == "__main__":
    create_icon() 