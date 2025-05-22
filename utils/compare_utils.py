from PIL import Image, ImageDraw, ImageFont

def create_collage(images, titles, output_path=None, images_per_row=5, padding=10, font_path=None, font_size=24):
    # 获取每行图片的最大宽度和最大高度
    max_img_width = max(img.width for img in images)
    max_img_height = max(img.height for img in images)
    
    # 计算拼接图片的总列数和行数
    total_images = len(images)
    rows = (total_images + images_per_row - 1) // images_per_row
    
    # 计算画布的宽度和高度
    collage_width = images_per_row * max_img_width + (images_per_row - 1) * padding
    collage_height = rows * (max_img_height + font_size + padding) + (rows - 1) * padding
    
    # 加载字体
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(font_size)
    
    # 创建空白画布
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
    draw = ImageDraw.Draw(collage)
    
    # 逐行逐列粘贴图片和标题
    x_offset = 0
    y_offset = 0
    for idx, (img, title) in enumerate(zip(images, titles)):
        # 使用 textbbox 获取标题的边界框尺寸
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_x = x_offset + (max_img_width - text_width) // 2

        draw.text((text_x, y_offset), title, fill=(0, 0, 0), font=font)
        
        # 计算图片位置
        img_y = y_offset + font_size + padding
        collage.paste(img, (x_offset, img_y))
        
        # 更新列偏移
        x_offset += max_img_width + padding
        
        # 每行到达 images_per_row 个图片后换行
        if (idx + 1) % images_per_row == 0:
            x_offset = 0
            y_offset += max_img_height + font_size + padding * 2
    
    # 保存拼接图片
    if output_path is not None:
        collage.save(output_path)
        print(f"拼接图片已保存到: {output_path}")
    return collage

def concatenate_images_with_caption(img1, img2, caption, output_path, padding=10, font_path=None, font_size=36):
    # 获取两张图片的宽度和高度
    width = max(img1.width, img2.width)
    height = img1.height + img2.height + padding

    # 加载字体
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(font_size)
    
    # 将多行 caption 拆分为行列表
    caption_lines = caption.split('\n')
    
    # 计算每行文字的高度，并计算整体高度
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in caption_lines)
    total_height = height + text_height + padding * (len(caption_lines) + 2)

    # 创建一个新的空白图像，宽度为两张图片的最大宽度，高度为两张图片高度之和加上文字说明
    new_image = Image.new('RGB', (width, total_height), (255, 255, 255))  # 使用白色背景
    
    # 将第一张图片粘贴到新图像的顶部
    new_image.paste(img1, (0, 0))
    
    # 将第二张图片粘贴到新图像的底部，Y位置为第一张图片的高度加上间隔
    new_image.paste(img2, (0, img1.height + padding))
    
    # 在图片底部逐行绘制文字说明
    draw = ImageDraw.Draw(new_image)
    y_text = height + padding
    for line in caption_lines:
        text_width = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, y_text), line, fill=(0, 0, 0), font=font)
        y_text += font_size + padding  # 为每行增加间距
    
    # 保存拼接后的图片
    new_image.save(output_path)
    print(f"拼接图片已保存到: {output_path}")

def create_collage_images_with_caption(img1, caption, output_path=None, padding=10, font_path=None, font_size=36):
    # 获取两张图片的宽度和高度
    width = img1.width
    height = img1.height + padding

    # 加载字体
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(font_size)
    
    # 将多行 caption 拆分为行列表
    caption_lines = caption.split('\n')
    
    # 计算每行文字的高度，并计算整体高度
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in caption_lines)
    total_height = height + text_height + padding * (len(caption_lines) + 2)

    # 创建一个新的空白图像，宽度为两张图片的最大宽度，高度为两张图片高度之和加上文字说明
    new_image = Image.new('RGB', (width, total_height), (255, 255, 255))  # 使用白色背景
    
    # 将第一张图片粘贴到新图像的顶部
    new_image.paste(img1, (0, 0))
    
    # 在图片底部逐行绘制文字说明
    draw = ImageDraw.Draw(new_image)
    y_text = height + padding
    for line in caption_lines:
        text_width = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, y_text), line, fill=(0, 0, 0), font=font)
        y_text += font_size + padding  # 为每行增加间距
    
    # 保存拼接后的图片
    if output_path is not None:
        new_image.save(output_path)
        print(f"图片已保存到: {output_path}")
    return new_image

# images = [Image.open(f"images/e{i}.jpg") for i in range(2, 14)]
# titles = [f"Title {i}" for i in range(2, 14)]

# create_collage(images, titles, "compare_output/collage_output.jpg")