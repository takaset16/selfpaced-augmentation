# coding: utf-8
import numpy as np
import cv2
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
import torch
import util


def random_noise(image):
    noise_scale = 0.001
    noise = np.random.randn(np.array(image.data).shape[0], np.array(image.data).shape[1], np.array(image.data).shape[2], np.array(image.data).shape[3])  # ノイズ生成

    image = image + util.to_var(torch.from_numpy(noise_scale * noise).float())

    # util.save_images(image)

    return image


def horizontal_flip(image):
    n, _, _, _ = image.shape
    image2 = image.clone()

    rand = np.random.rand(n)
    reverse = torch.arange(image.shape[3] - 1, -1, -1)

    for i in range(n):
        if rand[i] < 0.5:
            image2[i] = image[i, :, :, reverse]

    # util.save_images(image2)

    return image2


def vertical_flip(image):
    n, _, _, _ = image.shape
    image2 = image.clone()

    rand = np.random.rand(n)
    reverse = torch.arange(image.shape[2] - 1, -1, -1)

    for i in range(n):
        if rand[i] < 0.5:
            image2[i] = image[i, :, reverse, :]

    # util.save_images(image2)

    return image2


def random_crop(image):
    n, c, h, w = image.shape
    image2 = torch.zeros(n, c, h, w).cuda()

    # crop_size = (9 * h // 10, 9 * w // 10)
    crop_size = (4 * h // 5, 4 * w // 5)

    for i in range(n):
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])

        bottom = top + crop_size[0]
        right = left + crop_size[1]

        x = image[i, :, top:bottom, left:right].clone()
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])

        image2[i] = torch.nn.functional.upsample(x, size=(h, w), mode="bilinear", align_corners=True)

    # util.save_images(image)

    return image2


def random_transfer(image):
    n, c, h, w = image.shape
    image2 = torch.zeros(n, c, h, w).cuda()

    offset = np.random.randint(-h//5, h//5 + 1, size=(n, 2))
    # offset1 = int(h * 0.2)
    # offset2 = int(h * 0.2)

    for i in range(n):
        offset1, offset2 = offset[i]

        left = max(0, offset1)
        top = max(0, offset2)
        right = min(w, w + offset1)
        bottom = min(h, h + offset2)

        image2[i, :, top - offset2:bottom - offset2, left - offset1:right - offset1] = image[i, :, top:bottom, left:right]

    # util.save_images(image2)

    return image2


def random_rotation(image):
    size, c, h, w = image.shape
    image2 = image.clone()
    image2 = np.array(image2.data.cpu())

    if c == 1:
        image2 = np.squeeze(image2)
    else:
        image2 = image2.transpose((0, 2, 3, 1))

    for i in range(size):
        angle = np.random.randint(180)
        image_rotate = rotate(image2[i], angle)

        if c == 1:
            image2[i] = imresize(image_rotate, (h, w))
        else:
            image2[i] = imresize(image_rotate, (h, w, c))
    if c == 1:
        image2 = image2[:, np.newaxis, :, :] / 255.0
    else:
        image2 = image2.transpose((0, 3, 1, 2)) / 255.0

    image2 = torch.from_numpy(image2).float()
    # util.save_images(image2)

    return image2


def mixup(image, label, num_classes):
    alpha = 1.0

    rand_idx = torch.randperm(label.shape[0])
    image2 = image[rand_idx].clone()
    label2 = label[rand_idx].clone()

    y_one_hot = torch.eye(num_classes, device='cuda')[label].clone()
    y2_one_hot = torch.eye(num_classes, device='cuda')[label2].clone()
    mix_rate = np.random.beta(alpha, alpha, image.shape[0])

    mix_rate2 = None
    if image.ndim == 2:
        mix_rate2 = util.to_var(torch.from_numpy(mix_rate.reshape((image.shape[0], 1))).float())
    elif  image.ndim == 4:
        mix_rate2 = util.to_var(torch.from_numpy(mix_rate.reshape((image.shape[0], 1, 1, 1))).float())

    mix_rate = util.to_var(torch.from_numpy(mix_rate.reshape((image.shape[0], 1))).float())

    x_mixed = image.clone() * mix_rate2 + image2.clone() * (1 - mix_rate2)
    y_soft = y_one_hot * mix_rate + y2_one_hot * (1 - mix_rate)

    # util.save_images(x_mixed)

    return x_mixed, y_soft


def cutout(image):
    image2 = image.clone()
    n, _, h, w = image2.shape
    mask_size = h // 2

    for i in range(n):
        mask_value = image2.mean()
        top = np.random.randint(0 - mask_size // 2, h - mask_size // 2)
        left = np.random.randint(0 - mask_size // 2, w - mask_size // 2)
        bottom = top + mask_size
        right = left + mask_size

        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > h:
            bottom = h
        if right > w:
            right = w

        image2[i][:, top:bottom, left:right] = mask_value

    # util.save_images(image2)

    return image2


def random_erasing(image, p=0.5, s=(0.02, 0.4), r=(0.3, 3)):
    if np.random.rand() > p:
       return image

    image2 = image.clone()

    n, _, h, w = image2.shape

    for i in range(n):
        mask_value = np.random.rand()

        mask_area = np.random.randint(h * w * s[0], h * w * s[1])

        mask_aspect_ratio = np.random.rand() * r[1] + r[0]

        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
        if mask_height > h - 1:
            mask_height = h - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w - 1:
            mask_width = w - 1

        top = np.random.randint(0, h - mask_height)
        left = np.random.randint(0, w - mask_width)
        bottom = top + mask_height
        right = left + mask_width

        image2[i][:, top:bottom, left:right] = mask_value

    # util.save_images(image2)

    return image


def ricap(image_batch, label_batch, num_classes):
    image_batch = np.array(image_batch.data.cpu())
    label_batch = np.array(label_batch.data.cpu())

    label_batch = np.identity(num_classes)[label_batch]

    alpha = 1.0
    use_same_random_value_on_batch = False

    batch_size = image_batch.shape[0]
    image_y = image_batch.shape[2]
    image_x = image_batch.shape[3]

    # crop_size w, h from beta distribution
    if use_same_random_value_on_batch:
        w_dash = np.random.beta(alpha, alpha) * np.ones(batch_size)
        h_dash = np.random.beta(alpha, alpha) * np.ones(batch_size)
    else:
        w_dash = np.random.beta(alpha, alpha, size=batch_size)
        h_dash = np.random.beta(alpha, alpha, size=batch_size)
    w = np.round(w_dash * image_x).astype(np.int32)
    h = np.round(h_dash * image_y).astype(np.int32)

    output_images = np.zeros(image_batch.shape)
    output_labels = np.zeros(label_batch.shape)

    def create_masks(start_xs, start_ys, end_xs, end_ys):
        mask_x = np.logical_and(np.arange(image_x).reshape(1, 1, 1, -1) >= start_xs.reshape(-1, 1, 1, 1),
                                np.arange(image_x).reshape(1, 1, 1, -1) < end_xs.reshape(-1, 1, 1, 1))
        mask_y = np.logical_and(np.arange(image_y).reshape(1, 1, -1, 1) >= start_ys.reshape(-1, 1, 1, 1),
                                np.arange(image_y).reshape(1, 1, -1, 1) < end_ys.reshape(-1, 1, 1, 1))
        mask = np.logical_and(mask_y, mask_x)
        mask = np.logical_and(mask, np.repeat(True, image_batch.shape[1]).reshape(1, -1, 1, 1))

        return mask

    def crop_concatenate(wk, hk, start_x, start_y, end_x, end_y):
        nonlocal output_images, output_labels
        xk = (np.random.rand(batch_size) * (image_x - wk)).astype(np.int32)
        yk = (np.random.rand(batch_size) * (image_y - hk)).astype(np.int32)
        target_indices = np.arange(batch_size)
        np.random.shuffle(target_indices)
        weights = wk * hk / image_x / image_y

        dest_mask = create_masks(start_x, start_y, end_x, end_y)
        target_mask = create_masks(xk, yk, xk + wk, yk + hk)

        output_images[dest_mask] = image_batch[target_indices][target_mask]
        output_labels += weights.reshape(-1, 1) * label_batch[target_indices]

    crop_concatenate(w, h,
                     np.repeat(0, batch_size), np.repeat(0, batch_size),
                     w, h)
    crop_concatenate(image_x - w, h,
                     w, np.repeat(0, batch_size),
                     np.repeat(image_x, batch_size), h)
    crop_concatenate(w, image_y - h,
                     np.repeat(0, batch_size), h,
                     w, np.repeat(image_y, batch_size))
    crop_concatenate(image_x - w, image_y - h,
                     w, h, np.repeat(image_x, batch_size),
                     np.repeat(image_y, batch_size))

    output_images = torch.from_numpy(output_images).float()
    output_labels = torch.from_numpy(output_labels).float()
    label_batch = torch.from_numpy(label_batch).float()

    # util.save_images(output_images)

    return output_images, output_labels
