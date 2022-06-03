from PIL import Image


def img_to_pdf_v2(img_path_arr, dst_path):
    from fpdf import FPDF

    class MyPDF(FPDF):
        def footer(self):
            # Go to 1.5 cm from bottom
            self.set_y(0)
            # Select Arial italic 8
            # self.set_font('Arial', 'I', 8)
            # Print centered page number
            # self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    # fpdf = FPDF(orientation = 'P', unit = 'mm', format='A4')
    # pdf = FPDF(orientation='L', format='a5')
    # self.figsize = (10, 6)
    # or a tuple containing the width and the height
    img_w = 100  # 190
    img_h = 60
    pdf = MyPDF(orientation='P', unit='mm', format=(img_w + 10, img_h + 10))
    # pdf = FPDF(orientation='L', unit='mm', format='a5')
    # pdf = FPDF(orientation='L', unit='mm', format=(100+50, 160+39))
    # pdf = FPDF(orientation='L', unit='mm', format=(100 + 50, 160 + 39))
    # fpdf.set_margins(left: float, top: float, right: float = -1)
    # imagelist is the list with all image filenames
    for image in img_path_arr:
        # fpdf.add_page(orientation = '', format = '', same = False)
        pdf.add_page()
        pdf.set_margins(0, 0, 0)
        pdf.image(image, w=img_w)
        # pdf.set_left_margin(0)
        # pdf.set_right_margin(0)
        # pdf.image(image, x, y, w, h)

    pdf.output(dst_path, "F")


def img_to_pdf(img_path_arr, pdf_path):
    img_arr = []
    img_0 = None
    for img_path in img_path_arr:
        img = Image.open(img_path)
        # img.convert('RGB')

        png = img
        png.load()  # required for png.split()
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        img = background

        if img_0 is None:
            img_0 = img
        else:
            img_arr.append(img)

    # Load all the images in dedicated variable
    # image1 = Image.open('Flower Image 1.jpg')
    # image2 = Image.open('Flower Image 2.jpg')
    # image3 = Image.open('Flower Image 3.jpg')
    # image4 = Image.open('Flower Image 4.jpg')
    # image5 = Image.open('Flower Image 5.jpg')

    # Convert all the images to RGB
    # image1.convert('RGB')
    # image2.convert('RGB')
    # image3.convert('RGB')
    # image4.convert('RGB')
    # image5.convert('RGB')

    # List of image variables (without the first image)
    # Maintain image order if necessary
    # I'm not including my desired order's first image in the list
    # My desired first image is "image1"
    # imageList = [image2, image3, image4, image5]

    # Creation of PDF
    # pdf_path = 'Flowers.pdf'  # Filename of PDF
    # Now is the perfect time to use my first image
    # The PDF will organize Flower images in below order
    # image1, image2, image3, image4, image5
    # image1 is showing first because of using this for the saving process
    img_0.save(pdf_path, save_all=True, append_images=img_arr)

    # End
    print('Done')


if __name__ == '__main__':
    img_to_pdf()