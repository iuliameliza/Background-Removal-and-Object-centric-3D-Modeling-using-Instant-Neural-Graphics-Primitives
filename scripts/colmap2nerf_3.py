#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from glob import glob
import os
from pathlib import Path, PurePosixPath
from PIL import Image
from moviepy.editor import *
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
import time
import ffmpeg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS_FOLDER = os.path.join(ROOT_DIR, "scripts")

def parse_args():
	parser = argparse.ArgumentParser(description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

	parser.add_argument("--video_in", default="", help="Run ffmpeg first to convert a provided video file into a set of images. Uses the video_fps parameter also.")
	parser.add_argument("--video_fps", default=2)
	parser.add_argument("--time_slice", default="", help="Time (in seconds) in the format t1,t2 within which the images should be generated from the video. E.g.: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video.")
	parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
	parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
	parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
	parser.add_argument("--colmap_camera_params", default="", help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
	parser.add_argument("--images", default="images", help="Input path to the images.")
	parser.add_argument("--text", default="colmap_text", help="Input path to the colmap text files (set automatically if --run_colmap is used).")
	parser.add_argument("--aabb_scale", default=32, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128")
	parser.add_argument("--skip_early", default=0, help="Skip this many images from the start.")
	parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering).")
	parser.add_argument("--out", default="transforms.json", help="Output path.")
	parser.add_argument("--vocab_path", default="", help="Vocabulary tree path.")
	parser.add_argument("--overwrite", action="store_true", help="Do not ask for confirmation for overwriting existing images and COLMAP data.")
	parser.add_argument("--mask_categories", nargs="*", type=str, default=[], help="Object categories that should be masked out from the training images. See `scripts/category2id.json` for supported categories.")
	parser.add_argument("--bgrmv", type=int, default=0, help="What background removal algorithm to use (0/1)")
	parser.add_argument('--clean', action="store_true", help="Delete everything before starting again.")
	parser.add_argument('--resize', action="store_true", help="Resize the initial video to a lwoer quality to ensure a faster generation.")
	args = parser.parse_args()
	return args

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

def get_fps(video_path):
	cam = cv2.VideoCapture(video_path)
	
    # Find video length in seconds to compute the desired number of fps
	fps = cam.get(cv2.CAP_PROP_FPS)
	frame_count = cam.get(cv2.CAP_PROP_FRAME_COUNT)
	duration = round(frame_count / fps)

	return round(120/duration)

def reduce_video_size():
	# Get size of the video in MB
	size = "{:.2f}".format(os.path.getsize(args.video_in)/(1024*1024))

	# Optimal size for fast result is lower than 20 MB
	# Resizing the video if its size is bigger than 20 MB
	if float(size) > 20:
		# Resize the video to HD (1920:1080) using ffmpeg
		do_system(f"ffmpeg -i {args.video_in} -vf \"scale=\'if(gt(a,1920/1080),-1,1080):if(gt(a,1920/1080),1920,-1)\'\" -c:v libx264 -crf 23 resized.mp4")

		# Rename the initial video with a '_0' 
		os.rename(args.video_in, args.video_in.replace('.', '_0.'))

		# Rename the resized video to the name given as argument
		os.rename('resized.mp4', args.video_in)

def remove_background_pillow_folder(input_folder, output_folder, bg_type):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Set threshold and background color based on the background type
    if bg_type == 'white':
        threshold = (175, 175, 175)
    elif bg_type == 'black':
        threshold = (80, 80, 80)
    else:
        raise ValueError("Invalid background type. Only 'white' or 'black' are supported.")

    bg_color = (255, 255, 255, 0)

    # Iterate over the images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Read the input image
            input_path = os.path.join(input_folder, filename)
            image = Image.open(input_path).convert("RGBA")

            # Create a new image with transparent background
            bg_removed = Image.new("RGBA", image.size, bg_color)
        
            # Iterate over each pixel in the image
            for x in range(image.width):
                for y in range(image.height):
                    pixel = image.getpixel((x, y))

                    # Check if the pixel is below or above the threshold based on the background type and set it as transparent
                    if bg_type == 'white' and pixel[:3] > threshold:
                        bg_removed.putpixel((x, y), bg_color)
                    elif bg_type == 'black' and pixel[:3] < threshold:
                        bg_removed.putpixel((x, y), bg_color)
                    else:
                        bg_removed.putpixel((x, y), pixel)

            # Save the result in the output folder
            output_path = os.path.join(output_folder, filename.split('.')[0] + '.png')
            bg_removed.save(output_path)
	    
def remove_background_opencv_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Read the input image
            print(filename)
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)

            # Create a mask initialized with zeros
            mask = np.zeros(image.shape[:2], np.uint8)

            # Define the background and foreground model
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            # Define the region of interest (ROI) for the subject
            # You can modify these values based on the specific image and subject position
            rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

            # Apply GrabCut algorithm to segment the foreground from the background
            cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

            # Create a mask where the probable foreground and definite foreground are set as 1
            mask_fg = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Apply the mask to the original image to remove the background
            result = image * mask_fg[:, :, np.newaxis]

            # Save the result in the output folder
            output_path = os.path.join(output_folder, filename.split('.')[0] + '.png')
            cv2.imwrite(output_path, result)

def remove_background_scikit_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Read the input image
            filepath = os.path.join(input_folder, filename)

			# read the image in grayscale
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # apply sobel edge detection
            elevation_map = sobel(img)

            # create a circular mask for the undefined region
			# consider the undefined region as the center of the image
            markers = np.zeros_like(img)
            r, c = np.indices(img.shape)
            radius = min(img.shape) // 2
            center_radius = radius // 2
            central_pixels = (r - img.shape[0] // 2)**2 + (c - img.shape[1] // 2)**2 < center_radius**2
            external_pixels = (r - img.shape[0] // 2)**2 + (c - img.shape[1] // 2)**2 > radius**2
            markers[external_pixels] = 1
            markers[central_pixels] = 2

            # apply the watershed algorithm
            segmentation = watershed(elevation_map, markers)

            # create mask for the background
            background = (segmentation == 1)

            # read the original image
            original = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
	    
            # if original image is not rgba, convert it to rgba
            if original.shape[2] != 4:
                original = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)

            # apply the mask, set background to transparent
            result = np.where(background[..., None], [0, 0, 0, 0], original)

            # Save the result in the output folder
            output_path = os.path.join(output_folder, filename.split('.')[0] + '.png')
            cv2.imwrite(output_path, result)

def run_ffmpeg(args):
	if os.path.exists('images'):
		return
	
	ffmpeg_binary = "ffmpeg"

	# On Windows, if FFmpeg isn't found, try automatically downloading it from the internet
	if os.name == "nt" and os.system(f"where {ffmpeg_binary} >nul 2>nul") != 0:
		ffmpeg_glob = os.path.join(ROOT_DIR, "external", "ffmpeg", "*", "bin", "ffmpeg.exe")
		candidates = glob(ffmpeg_glob)
		if not candidates:
			print("FFmpeg not found. Attempting to download FFmpeg from the internet.")
			do_system(os.path.join(SCRIPTS_FOLDER, "download_ffmpeg.bat"))
			candidates = glob(ffmpeg_glob)

		if candidates:
			ffmpeg_binary = candidates[0]

	if not os.path.isabs(args.images):
		args.images = os.path.join(os.path.dirname(args.video_in), args.images)

	if args.resize:
		reduce_video_size()

	images = "\"" + args.images + "\""
	video =  "\"" + args.video_in + "\""
	# fps = float(args.video_fps) or 1.0
	fps = get_fps(args.video_in)
	print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
	try:
		# Passing Images' Path Without Double Quotes
		shutil.rmtree(args.images)
	except:
		pass
	do_system(f"mkdir images1")

	time_slice_value = ""
	time_slice = args.time_slice
	if time_slice:
		start, end = time_slice.split(",")
		time_slice_value = f",select='between(t\,{start}\,{end})'"
	do_system(f"{ffmpeg_binary} -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" images1/%04d.jpg")

	if args.bgrmv == 0:
		do_system('rembg p images1 ' + 'images')
	elif args.bgrmv == 1:
		do_system(f'mkdir images')
		for img in os.listdir('images1'):
			command = 'backgroundremover -i images1\\' + img + ' -o images\\' + img.split('.')[0] + '.png'
			do_system(command)
	elif args.bgrmv == 2:
		remove_background_opencv_folder('images1', 'images')
	elif args.bgrmv == 3:
		remove_background_scikit_folder('images1', 'images')
	elif args.bgrmv == 4:
		remove_background_pillow_folder('images1', 'images', 'white')
	elif args.bgrmv == 5:
		remove_background_pillow_folder('images1', 'images', 'black')

	try:
		shutil.rmtree('images1')
		# shutil.rmtree('images2')
	except:
		pass

def clean():
	if len(os.listdir(os.getcwd())) == 1:
		return
	for file in os.listdir(os.getcwd()):
		if file == args.video_in.replace('.', '_0.'):
			os.rename(file, args.video_in)
			continue
		elif os.path.isdir(file):
			try:
				shutil.rmtree(file)
			except:
				pass
		elif os.path.isfile(file):
			try:
				os.remove(file)
			except:
				pass

def run_colmap(args):
	colmap_binary = "colmap"

	# On Windows, if FFmpeg isn't found, try automatically downloading it from the internet
	if os.name == "nt" and os.system(f"where {colmap_binary} >nul 2>nul") != 0:
		colmap_glob = os.path.join(ROOT_DIR, "external", "colmap", "*", "COLMAP.bat")
		candidates = glob(colmap_glob)
		if not candidates:
			print("COLMAP not found. Attempting to download COLMAP from the internet.")
			do_system(os.path.join(SCRIPTS_FOLDER, "download_colmap.bat"))
			candidates = glob(colmap_glob)

		if candidates:
			colmap_binary = candidates[0]

	db = args.colmap_db
	images = "\"" + args.images + "\""
	db_noext=str(Path(db).with_suffix(""))

	if args.text=="text":
		args.text=db_noext+"_text"
	text=args.text
	sparse=db_noext+"_sparse"
	print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
	# if not args.overwrite and (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
	# 	sys.exit(1)
	if os.path.exists(db):
		os.remove(db)
	do_system(f"{colmap_binary} feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params \"{args.colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
	match_cmd = f"{colmap_binary} {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"
	if args.vocab_path:
		match_cmd += f" --VocabTreeMatching.vocab_tree_path {args.vocab_path}"
	do_system(match_cmd)
	try:
		shutil.rmtree(sparse)
	except:
		pass
	do_system(f"mkdir {sparse}")
	do_system(f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {sparse}")
	do_system(f"{colmap_binary} bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
	try:
		shutil.rmtree(text)
	except:
		pass
	do_system(f"mkdir {text}")
	do_system(f"{colmap_binary} model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

if __name__ == "__main__":
	start_time = time.time()
	args = parse_args()
	if args.bgrmv > 5:
		print('Unavailable option, please choose between 0 and 1 for background removal.')
		sys.exit(1)
	if args.clean:
		clean()
	if args.video_in != "":
		run_ffmpeg(args)
	if args.run_colmap:
		run_colmap(args)
	AABB_SCALE = int(args.aabb_scale)
	SKIP_EARLY = int(args.skip_early)
	IMAGE_FOLDER = args.images
	TEXT_FOLDER = args.text
	OUT_PATH = args.out
	print(f"outputting to {OUT_PATH}...")
	with open(os.path.join(TEXT_FOLDER,"cameras.txt"), "r") as f:
		angle_x = math.pi / 2
		for line in f:
			# 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
			# 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
			# 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
			if line[0] == "#":
				continue
			els = line.split(" ")
			w = float(els[2])
			h = float(els[3])
			fl_x = float(els[4])
			fl_y = float(els[4])
			k1 = 0
			k2 = 0
			k3 = 0
			k4 = 0
			p1 = 0
			p2 = 0
			cx = w / 2
			cy = h / 2
			is_fisheye = False
			if els[1] == "SIMPLE_PINHOLE":
				cx = float(els[5])
				cy = float(els[6])
			elif els[1] == "PINHOLE":
				fl_y = float(els[5])
				cx = float(els[6])
				cy = float(els[7])
			elif els[1] == "SIMPLE_RADIAL":
				cx = float(els[5])
				cy = float(els[6])
				k1 = float(els[7])
			elif els[1] == "RADIAL":
				cx = float(els[5])
				cy = float(els[6])
				k1 = float(els[7])
				k2 = float(els[8])
			elif els[1] == "OPENCV":
				fl_y = float(els[5])
				cx = float(els[6])
				cy = float(els[7])
				k1 = float(els[8])
				k2 = float(els[9])
				p1 = float(els[10])
				p2 = float(els[11])
			elif els[1] == "SIMPLE_RADIAL_FISHEYE":
				is_fisheye = True
				cx = float(els[5])
				cy = float(els[6])
				k1 = float(els[7])
			elif els[1] == "RADIAL_FISHEYE":
				is_fisheye = True
				cx = float(els[5])
				cy = float(els[6])
				k1 = float(els[7])
				k2 = float(els[8])
			elif els[1] == "OPENCV_FISHEYE":
				is_fisheye = True
				fl_y = float(els[5])
				cx = float(els[6])
				cy = float(els[7])
				k1 = float(els[8])
				k2 = float(els[9])
				k3 = float(els[10])
				k4 = float(els[11])
			else:
				print("Unknown camera model ", els[1])
			# fl = 0.5 * w / tan(0.5 * angle_x);
			angle_x = math.atan(w / (fl_x * 2)) * 2
			angle_y = math.atan(h / (fl_y * 2)) * 2
			fovx = angle_x * 180 / math.pi
			fovy = angle_y * 180 / math.pi

	print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

	with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
		i = 0
		bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
		out = {
			"camera_angle_x": angle_x,
			"camera_angle_y": angle_y,
			"fl_x": fl_x,
			"fl_y": fl_y,
			"k1": k1,
			"k2": k2,
			"k3": k3,
			"k4": k4,
			"p1": p1,
			"p2": p2,
			"is_fisheye": is_fisheye,
			"cx": cx,
			"cy": cy,
			"w": w,
			"h": h,
			"aabb_scale": AABB_SCALE,
			"frames": [],
		}

		up = np.zeros(3)
		for line in f:
			line = line.strip()
			if line[0] == "#":
				continue
			i = i + 1
			if i < SKIP_EARLY*2:
				continue
			if  i % 2 == 1:
				elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
				#name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
				# why is this requireing a relitive path while using ^
				image_rel = os.path.relpath(IMAGE_FOLDER)
				name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
				b = sharpness(name)
				print(name, "sharpness=",b)
				image_id = int(elems[0])
				qvec = np.array(tuple(map(float, elems[1:5])))
				tvec = np.array(tuple(map(float, elems[5:8])))
				R = qvec2rotmat(-qvec)
				t = tvec.reshape([3,1])
				m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
				c2w = np.linalg.inv(m)
				if not args.keep_colmap_coords:
					c2w[0:3,2] *= -1 # flip the y and z axis
					c2w[0:3,1] *= -1
					c2w = c2w[[1,0,2,3],:]
					c2w[2,:] *= -1 # flip whole world upside down

					up += c2w[0:3,1]

				frame = {"file_path":name,"sharpness":b,"transform_matrix": c2w}
				out["frames"].append(frame)
	nframes = len(out["frames"])

	if args.keep_colmap_coords:
		flip_mat = np.array([
			[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]
		])

		for f in out["frames"]:
			f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
	else:
		# don't keep colmap coords - reorient the scene to be easier to work with

		up = up / np.linalg.norm(up)
		print("up vector was", up)
		R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
		R = np.pad(R,[0,1])
		R[-1, -1] = 1

		for f in out["frames"]:
			f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

		# find a central point they are all looking at
		print("computing center of attention...")
		totw = 0.0
		totp = np.array([0.0, 0.0, 0.0])
		for f in out["frames"]:
			mf = f["transform_matrix"][0:3,:]
			for g in out["frames"]:
				mg = g["transform_matrix"][0:3,:]
				p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
				if w > 0.00001:
					totp += p*w
					totw += w
		if totw > 0.0:
			totp /= totw
		print(totp) # the cameras are looking at totp
		for f in out["frames"]:
			f["transform_matrix"][0:3,3] -= totp

		avglen = 0.
		for f in out["frames"]:
			avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
		avglen /= nframes
		print("avg camera distance from origin", avglen)
		for f in out["frames"]:
			f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

	for f in out["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
	print(nframes,"frames")
	print(f"writing {OUT_PATH}")
	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)

	if len(args.mask_categories) > 0:
		# Check if detectron2 is installed. If not, install it.
		try:
			import detectron2
		except ModuleNotFoundError:
			try:
				import torch
			except ModuleNotFoundError:
				print("PyTorch is not installed. For automatic masking, install PyTorch from https://pytorch.org/")
				sys.exit(1)

			input("Detectron2 is not installed. Press enter to install it.")
			import subprocess
			package = 'git+https://github.com/facebookresearch/detectron2.git'
			subprocess.check_call([sys.executable, "-m", "pip", "install", package])
			import detectron2

		import torch
		from pathlib import Path
		from detectron2.config import get_cfg
		from detectron2 import model_zoo
		from detectron2.engine import DefaultPredictor

		category2id = json.load(open(SCRIPTS_FOLDER / "category2id.json", "r"))
		mask_ids = [category2id[c] for c in args.mask_categories]

		cfg = get_cfg()
		# Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
		cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
		# Find a model from detectron2's model zoo.
		cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		predictor = DefaultPredictor(cfg)

		for frame in out['frames']:
			img = cv2.imread(frame['file_path'])
			outputs = predictor(img)

			output_mask = np.zeros((img.shape[0], img.shape[1]))
			for i in range(len(outputs['instances'])):
				if outputs['instances'][i].pred_classes.cpu().numpy()[0] in mask_ids:
					pred_mask = outputs['instances'][i].pred_masks.cpu().numpy()[0]
					output_mask = np.logical_or(output_mask, pred_mask)

			rgb_path = Path(frame['file_path'])
			mask_name = str(rgb_path.parents[0] / Path('dynamic_mask_' + rgb_path.name.replace('.jpg', '.png')))
			cv2.imwrite(mask_name, (output_mask*255).astype(np.uint8))

	end_time = time.time()
	total = end_time - start_time
	mins = int(total / 60)
	sec = int(total - (mins * 60))
	print('\n\nTotal time: ', mins , 'm ', sec, 's')