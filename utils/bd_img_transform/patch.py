import logging
import cv2
import numpy as np
import torch

class AddPatchTrigger(object):
    '''
    assume init use HWC format
    but in add_trigger, you can input tensor/array , one/batch
    '''
    def __init__(self, trigger_loc, trigger_ptn):
        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if isinstance(img, np.ndarray):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[m, n, :] = self.trigger_ptn[i]  # add trigger
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n, :] = self.trigger_ptn[i]  # add trigger
        elif isinstance(img, torch.Tensor):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n] = self.trigger_ptn[i]
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, :, m, n] = self.trigger_ptn[i]
        return img

import numpy as np

class AddMatrixPatchTrigger(object):
    '''
    tensor version of add trigger, this should be put after
    '''
    def __init__(self, trigger_tensor: np.ndarray, resize_trigger_check = True):
        logging.warning('Use cv2 resize in case non-same trigger and img blend. so torch tensor will fail')
        self.trigger_tensor = np.clip(trigger_tensor, 0, 255)  # notice that non-trigger parts must be zero !
        self.resize_trigger_check = resize_trigger_check

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        # two case, trigger and img shape same / not same
        try:
            after_blend_img = img * (self.trigger_tensor == 0) + self.trigger_tensor * (self.trigger_tensor > 0)
        except:
            if self.resize_trigger_check == True:
                print('NOT SAME size for img and trigger_tensor, use cv2 resize trigger_tensor')
                trigger_tensor = cv2.resize(self.trigger_tensor, dsize=(img.shape[:2])[::-1] if len(img.shape) <= 3 else img.shape[1:3][::-1])
                after_blend_img = img * (trigger_tensor == 0) + trigger_tensor * (trigger_tensor > 0)
            else:
                print('NOT SAME size for img and trigger_tensor, use cv2 resize img')
                img = cv2.resize(img,
                                            dsize=(self.trigger_tensor.shape[:2])[::-1] if len(self.trigger_tensor.shape) <= 3 else self.trigger_tensor.shape[1:3][::-1])
                after_blend_img = img * (self.trigger_tensor == 0) + self.trigger_tensor * (self.trigger_tensor > 0)

        return after_blend_img

from random import randint

class AddRandomColorTrigger_RandomLocEverytime(object):

    def __init__(self, trigger_loc, trigger_ptn, picsize_x, picsize_y):

        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn

        loc_x = [x for x,y in trigger_loc]
        loc_y = [y for x,y in trigger_loc]

        self.min_x = min(loc_x)
        self.min_y = min(loc_y)

        self.size_x = max(loc_x) - min(loc_x) + 1
        self.size_y = max(loc_x) - min(loc_x) + 1

        self.picsize_x = picsize_x
        self.picsize_y = picsize_y

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):

        self.random_shift_x = randint(0, max(self.picsize_x - self.size_x, 0))
        self.random_shift_y = randint(0, max(self.picsize_y - self.size_y, 0))
        # print(self.random_shift_x, self.random_shift_y)

        for i, (m, n) in enumerate(self.trigger_loc):
            # print(i, (m, n))

            m, n = m + self.random_shift_x - self.min_x, n + self.random_shift_y - self.min_y

            img[m, n, :] = self.trigger_ptn[i]  # add trigger

        return img