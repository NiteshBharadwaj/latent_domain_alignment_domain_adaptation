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
									img1 = img
									if (x1 != x2 and y1 != y2):
										img1 = img.crop((int(x1), int(y1), int(x2), int(y2)))
									img1.save(os.path.join(out_path, folder, folder2, folder3)+'/'+file_name+'.jpg')


def main():
	a = ImageCropper('./data_cropped')
	a.start_Cropping()

if __name__ == '__main__':
    main()