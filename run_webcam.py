from __future__ import print_function
import os
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time
import cv2
import chainer
from chainer import cuda, Variable, serializers
from net import *
import alsaaudio
import json


parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('--cam_device_id', '-v', default=-1, type=int, help='Device Id for webcam (negative value indicates select first available)')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--median_filter', default=3, type=int)
parser.add_argument('--padding', default=50, type=int)
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--disp_width', default=720, type=int)
parser.add_argument('--disp_source', action='store_true')
parser.add_argument('--horizontal', action='store_true')
parser.add_argument('--output_left', action='store_true')
parser.add_argument('--keep_colors', action='store_true')
parser.add_argument('--cycle', default=0, type=int)
parser.add_argument('--delay_next_model', default=1, type=int)
parser.add_argument('--trim_frame', default=0, type=int)
parser.add_argument('--screen_h', default=2160, type=int)
parser.add_argument('--screen_w', default=3840, type=int)
parser.add_argument('--border_h', default=0, type=int)
parser.add_argument('--border_w', default=0, type=int)


parser.set_defaults(keep_colors=False)
args = parser.parse_args()

models=[{"ckpt":"models/composition.model", "style":"sample_images/style_1.png"},
		{"ckpt":"models/cubist.model", "style":"sample_images/cubist-style.jpg"},
		{"ckpt":"models/edtaonisl.model", "style":"sample_images/edtaonisl.jpg"},
		{"ckpt":"models/kandinsky_e2_full512.model", "style":"sample_images/kandinsky.jpg"},
		{"ckpt":"models/seurat.model", "style":"sample_images/seurat.png"},
		{"ckpt":"models/candy_512_2_49000.model", "style":"sample_images/candy-style.jpg"},
		{"ckpt":"models/fur_0.model", "style":"sample_images/fur-style.jpg"},
		{"ckpt":"models/kanagawa.model", "style":"sample_images/kanagawa-style.jpg"},
		{"ckpt":"models/scream-style.model", "style":"sample_images/scream-style.jpg"},
		{"ckpt":"models/starry.model", "style":"sample_images/starry-style.jpg"}]

idx_model = 0


def get_camera_shape(cam):
	""" use a different syntax to get video size in OpenCV 2 and OpenCV 3 """
	cv_version_major, _, _ = cv2.__version__.split('.')
	if cv_version_major == '3' or cv_version_major == '4':
		return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	else:
		return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

  
# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')


def make_triptych(disp_width, frame, style, output, trim_frame=0, horizontal=True, output_left=False):
	ttop, tbottom = int(0.33 * trim_frame), int(0.67 * trim_frame)
	frame = frame[:, tbottom:-ttop, :]
	ch, cw, _ = frame.shape
	sh, sw, _ = style.shape
	oh, ow, _ = output.shape
	disp_height = int(disp_width * oh / ow)
	h = int(ch * disp_width * 0.5 / cw)
	w = int(cw * disp_height * 0.5 / ch)
	
	if horizontal:
		w2, h2 = w, 0.5 * disp_height
	else:
		w2, h2 = 0.5 * disp_width, h
		
	sx2, sy2, sw2, sh2 = 0, 0, sw, sh
	fdh = disp_height if horizontal else disp_width * oh // ow
	axis = 0 if horizontal else 1
	if float(w2)/h2 > float(sw2)/sh2:
		sh2 = sw2 * (float(h2) / w2)
		sy2 = 0.5 * (sh - sh2)
	else:
		sw2 = sh2 * (float(w2) / h2)
		sx2 = 0.5 * (sw - sw2)
	
	frame = cv2.resize(frame, (int(w2), int(h2)))
	style = cv2.resize(style[int(sy2):int(sy2+sh2), int(sx2):int(sx2+sw2), :], (int(w2), int(h2)))
	output = cv2.resize(output, (disp_width, fdh))
	frame_style = np.concatenate([frame, style], axis=axis)	
	full_img = np.concatenate([output, frame_style] if output_left else [frame_style, output], axis=1-axis)
	return full_img


def trim_triptych(full_img, sh, sw, border_h=0, border_w=0):
	fh, fw, _ = full_img.shape
	trim_t, trim_b, trim_l, trim_r = border_h, border_h, border_w, border_w
	mh, mw = float(sh - trim_t - trim_b) / fh, float(sw - trim_l - trim_r) / fw
	m = min(mh, mw)
	fh2, fw2 = int(fh * m), int(fw * m)
	full_img = cv2.resize(full_img, (fw2, fh2), interpolation = cv2.INTER_NEAREST)
	full_canvas = np.zeros((sh, sw, 3)).astype(np.uint8)
	fy2, fx2 = int(0.5 * (sh - fh2)), int(0.5 * (sw - fw2))
	full_canvas[fy2:fy2+fh2, fx2:fx2+fw2, :] = full_img
	return full_canvas
	

def load_model(idx_model):
	print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
	model_path, style_path = models[idx_model]['ckpt'], models[idx_model]['style']
	model = FastStyleNet()
	style = cv2.imread(style_path)
	style = cv2.transpose(style)
	style = cv2.flip(style, flipCode=1)
	serializers.load_npz(model_path, model)
	if args.gpu >= 0:
		cuda.get_device(args.gpu).use()
		model.to_gpu()
	xp = np if args.gpu < 0 else cuda.cupy
	return model, style, xp

def prev_model():
	global idx_model
	idx_model = (idx_model + len(models) - 1) % len(models)		
	return load_model(idx_model)

def next_model():
	global idx_model
	idx_model = (idx_model + 1) % len(models)
	return load_model(idx_model)
	

idx_model = 0
disp_source = args.disp_source
disp_width = args.disp_width
width = args.width
output_left = args.output_left
horizontal = args.horizontal
trim_frame = args.trim_frame
screen_h, screen_w = args.screen_h, args.screen_w 
border_h, border_w = args.border_h, args.border_w 
monitor_trans_x = 0 


model, style, xp = load_model(idx_model)
cam = cv2.VideoCapture(args.cam_device_id)

cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("frame", monitor_trans_x, 0)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)

cam_width, cam_height = get_camera_shape(cam)
height = int(cam_height * width / cam_width)

t1 = time.time()
lastvol = alsaaudio.Mixer().getvolume()[0]
log_file_name = "templog_%s"%(str(t1))


while True:
	ret, frame_orig = cam.read()

	frame_orig = cv2.flip(frame_orig, 1)
	frame = cv2.resize(np.array(frame_orig), (width, height))
	image = np.asarray(frame, dtype=np.float32).transpose(2, 0, 1)
	image = image.reshape((1,) + image.shape)

	if args.padding > 0:
		image = np.pad(image, [[0, 0], [0, 0], [args.padding, args.padding], [args.padding, args.padding]], 'symmetric')

	image = xp.asarray(image)
	x = Variable(image)
	y = model(x)
	result = cuda.to_cpu(y.data)

	if args.padding > 0:
		result = result[:, :, args.padding:-args.padding, args.padding:-args.padding]

	result = np.uint8(result[0].transpose((1, 2, 0)))
	result = result[:,:,[2,1,0]]

	med = Image.fromarray(result)
	if args.median_filter > 0:
		med = med.filter(ImageFilter.MedianFilter(args.median_filter))
	if args.keep_colors:
		med = original_colors(original, med)

	if disp_source:
		full_img = make_triptych(disp_width, frame, style, result, trim_frame, horizontal, output_left)
		full_img = trim_triptych(full_img, screen_h, screen_w, border_h, border_w)
		cv2.imshow('frame', full_img)
	else:
		oh, ow, _ = result.shape
		result = cv2.resize(result, (disp_width, int(oh * disp_width / ow)))
		cv2.imshow('frame', result)
		
	vol = alsaaudio.Mixer().getvolume()[0]
	vol_up, vol_down = vol > lastvol, vol < lastvol
	lastvol = vol
	if vol == 100 or vol == 0:
		alsaaudio.Mixer().setvolume(50)
		lastvol = 50

	t2 = time.time()
	dt = t2 - t1
	key_ = cv2.waitKey(1)   
	
	if (key_ == ord('a') or vol_down) and dt > args.delay_next_model:
		t1 = t2
		model, style, xp = prev_model()

	elif (key_ == ord('s') or vol_up) and dt > args.delay_next_model:
		t1 = t2
		model, style, xp = next_model()
	
	elif key_ == 27:
		break

	if args.cycle > 0:
		t2 = time.time()
		dt = t2 - t1
		if dt > args.cycle:
			t1 = t2
			model, style, xp = next_model()


# done
cam.release()
cv2.destroyAllWindows()

