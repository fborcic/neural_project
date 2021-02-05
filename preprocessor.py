import itertools
import sys

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt
from collections import Counter

# standard weights for RGB to grayscale conversion
GRAYSCALE_WEIGHTS = [0.2989, 0.5870, 0.1140]

# thresholding (binarization) parameters - explained in ppt
THRESHOLD_SPACE = (.2, .8, 50) # threshold space
ALLOWED_BLACK_IN_BG = 0.015 # ratio of black pixels in vertical extremes
BG_LINES = 5 # extent of vertical extremes

TRIM_PADDING = .1 # how much background to leave when trimming

MIN_DIGIT_AREA = 100 # minimum number of pixels that can constitute a digit
OVERLAP_THRESHOLD = .7 # horizontal overlap of contiguous areas to be joined

# helper list for flood-fill
NEIGHBORS = [c for c in itertools.product([0,1,-1], repeat=2) if c!=(0, 0)]

DEBUG_PREFIX = 'debug'

class InputPreprocessor(object):
    """ This class takes a numpy array representing an RGB image in its constructor,
        and separates out the digits.

        If debug_name is provided, it will save the intermediate processing results
        to the current working directory"""
    def __init__(self, image, debug_name=None):
        self.input = image
        self.working_copy = image.copy()
        self._dbg = debug_name

        self.bounding_boxes = None # bounding boxes of digits
        self.cc_map = None # connected components map
        self.digits = [] # list of found digits, final result

    def _dbg_save(self, part, image=None):
        """ This method takes care of the debug save """
        if self._dbg is None:
            return

        if image is None:
            arr = self.working_copy
        else:
            arr = image

        arr = (arr*255).astype('uint8')
        Image.fromarray(arr).save('%s-%s.png'%(self._dbg, part))

    def _to_grayscale(self):
        """ This method converts the image to grayscale """
        self.working_copy = sum(factor*self.working_copy[:,:,idx] \
                for idx, factor in enumerate(GRAYSCALE_WEIGHTS))
        self.working_copy /= self.working_copy.max()
        self._dbg_save('grayscale')

    def _find_threshold(self):
        """ This method finds the adequate binarization threshold """
        optimal = 0
        img = self.working_copy
        for thr in np.linspace(*THRESHOLD_SPACE):
            black_count = (img[-BG_LINES:] < thr).sum() + (img[:BG_LINES] < thr).sum()
            if black_count/(2*BG_LINES*img.shape[0]) < ALLOWED_BLACK_IN_BG:
                optimal = thr
            else:
                break
        return optimal

    def _to_binary(self):
        """ This method performs thresholding (binarization) of the grayscale image """
        self.working_copy = (self.working_copy > self._find_threshold()).astype('float')
        self._dbg_save('binary')

    def _trim_likely_background(self):
        """ This method removes vertical whitespace to ease further processing """
        img = self.working_copy
        vert_dist = img.sum(axis=1)
        peak = np.argmin(vert_dist)
        tolerable = img.shape[1]*(1-ALLOWED_BLACK_IN_BG)
        lower_boundary = next(i for i in range(peak, -1, -1)
                if vert_dist[i] >= tolerable or i == 0)
        upper_boundary = next(i for i in range(peak, img.shape[0])
                if vert_dist[i] >= tolerable or i == img.shape[0])

        padding = int(TRIM_PADDING*img.shape[0])
        self.working_copy = img[lower_boundary-padding:upper_boundary+padding,:]
        self._dbg_save('background_trim')

    def _mark_bounding_boxes(self):
        """ This method marks the bounding boxes of digits for the purposes
            of the debug save """
        if self._dbg is None:
            return

        img = self.working_copy.copy()
        for bbox in self.bounding_boxes:
            x0, y0, x1, y1 = bbox[1]
            img[y0,x0:x1+1] = 0.5
            img[y1,x0:x1+1] = 0.5
            img[y0:y1+1,x0] = 0.5
            img[y0:y1+1,x1] = 0.5
        return img

    def _get_bounding_boxes(self):
        """ This method finds connected components in the image and saves their map,
        as well as the bounding boxes """
        img = self.working_copy.copy()
        blacks = set(zip(*np.where(img==0)))
        current_label = 1
        bounding_boxes = []

        while blacks: # flood fill loop
            initial = blacks.pop()
            to_consider = set([initial])
            current_label += 1
            while to_consider:
                pix = to_consider.pop()
                img[pix] = current_label
                if pix in blacks: blacks.remove(pix)
                y, x = pix
                for dx, dy in NEIGHBORS:
                    newpix = y+dy, x+dx
                    if (0 <= x+dx < img.shape[1] and
                            0 <= y+dy < img.shape[0] and
                            img[newpix] == 0):
                        to_consider.add(newpix)

        for label in range(2, current_label+1):
            mask = (img == label)
            ys, xs = np.where(mask)
            bounding_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        self.bounding_boxes = list(enumerate(bounding_boxes))
        self.cc_map = img-2 # this is to reconcile labels with the enumerate above
        self._dbg_save('bounding', self._mark_bounding_boxes())

    def _filter_bounding_boxes(self):
        """ This method removes components that are too small and tries to join disconnected
        parts of the digits (if there are any), by matching their ranges on the x axis """
        for bbox in self.bounding_boxes.copy():
            x0, y0, x1, y1 = bbox[1]
            if (x1-x0)*(y1-y0) < MIN_DIGIT_AREA: # if it's too small to be a digit
                self.bounding_boxes.remove(bbox)

        pointer = 0
        label = self.cc_map.max()+1
        while pointer < len(self.bounding_boxes)-1: # inefficient, but works on very little data
            self.bounding_boxes.sort(key=lambda b: b[1][0]+b[1][1]) # sort by x of centers
            bbox_a, bbox_b = self.bounding_boxes[pointer:pointer+2]
            x0a, y0a, x1a, y1a = bbox_a[1]
            x0b, y0b, x1b, y1b = bbox_b[1]
            len_a = x1a - x0a
            len_b = x1b - x0b
            len_intersect = max(0, min(x1a, x1b)-max(x0a, x0b))
            if len_intersect/min(len_a, len_b) > OVERLAP_THRESHOLD: # if the x overlap is large
                # ...join boxes
                x0, y0 = min(x0a, x0b), min(y0a, y0b)
                x1, y1 = max(x1a, x1b), max(y1a, y1b)
                self.cc_map[np.isin(self.cc_map, [bbox_a[0], bbox_b[0]])] = label
                for b in bbox_a, bbox_b:
                    self.bounding_boxes.remove(b)
                self.bounding_boxes.append((label, (x0, y0, x1, y1)))
                label += 1
                pointer = 0 # start over
            else:
                pointer += 1 # no change, go further

        self._dbg_save('bounding_filtered', self._mark_bounding_boxes())

    def _crop_digits(self):
        """ This method crops digits from the connected components map """
        for bbox in self.bounding_boxes:
            label, coords = bbox
            x0, y0, x1, y1 = coords
            mask = (self.cc_map == label).astype('float') # here the colors are inverted
            digit = mask[y0:y1+1,x0:x1+1]
            self.digits.append(digit)

        if self._dbg is not None:
            for n, digit in enumerate(self.digits):
                self._dbg_save('digit%d'%n, digit)

    def process(self):
        self._to_grayscale()
        self._to_binary()
        self._trim_likely_background()
        self._get_bounding_boxes()
        self._filter_bounding_boxes()
        self._crop_digits()

def scale_to_fit(image, padding=1, size=(28,28)):
    """ This functions uses facilities provided by PIL to scale and antialias
        the digits. """
    w, h = image.size
    w_exp, h_exp = size[0]-2*padding, size[1]-2*padding

    if abs(w-w_exp) > abs(h-h_exp):
        newsize = (w_exp, min(int(h*w_exp/w), h_exp))
    else:
        newsize = (min(int(w*h_exp/h), w_exp), h_exp)

    dx = (size[0]-newsize[0])//2
    dy = (size[1]-newsize[1])//2

    scaled = image.resize(newsize, Image.ANTIALIAS)
    out = Image.new(scaled.mode, size, 0)
    out.paste(scaled, (dx, dy))
    return out

# all further code is just for debugging purposes
if __name__ == '__main__':
    img = Image.open(sys.argv[1])
    input_image = InputPreprocessor(np.asarray(img).astype('float')/255, DEBUG_PREFIX)
    input_image.process()

    pil_digits = []
    for digit in input_image.digits:
        pil_digit_insert = Image.fromarray((digit*255).astype('uint8'))
        pil_digits.append(scale_to_fit(pil_digit_insert))

    for n,d in enumerate(pil_digits):
        d.save('dig%d.png'%n)
