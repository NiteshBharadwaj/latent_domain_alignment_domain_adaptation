from PIL import Image
import numpy as np
import os

base_dir='./data'

class ImageCropper():
	def __init__(self, out_dir):
		self.out_dir = out_dir

	def start_Cropping(self):
		base_path = os.path.join(base_dir,'CCWeb','data')
		out_path = os.path.join(self.out_dir,'CCWeb','data','image')

		base_path_labels = os.path.join(base_path, 'label')
		base_path_images = os.path.join(base_path, 'image')

		for folder in os.listdir(base_path_labels):
			for folder2 in os.listdir(os.path.join(base_path_labels, folder)):
				for folder3 in os.listdir(os.path.join(base_path_labels, folder, folder2)):
					for label_file in os.listdir(os.path.join(base_path_labels, folder, folder2, folder3)):
						file_name = label_file.split('.')[0]

						img_file = os.path.join(base_path_images, folder, folder2, folder3)+'/'+file_name+'.jpg'
						print('FileName', img_file)
						img = Image.open(img_file)
						
						os.makedirs(os.path.join(out_path, folder, folder2, folder3), exist_ok=True)

						with open(os.path.join(base_path_labels, folder, folder2, folder3)+'/'+label_file) as f:
							cnt = 0
							for line in f:
								cnt += 1
								if (cnt == 3):
									x1,y1,x2,y2 = line.split(' ')
									x1 = int(x1)
									y1 = int(y1)
									x2 = int(x2)
									y2 = int(y2)
									img1 = img

									x_max, y_max = img.size

									if (x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0 and x1 < x_max and x2 < x_max and y1 < y_max and y2 < y_max and x1 != x2 and y1 != y2):
										img1 = img.crop((x1, y1, x2, y2))
									img1.save(os.path.join(out_path, folder, folder2, folder3)+'/'+file_name+'.jpg')


def main():
	a = ImageCropper('./data_cropped')
	a.start_Cropping()

if __name__ == '__main__':
    main()