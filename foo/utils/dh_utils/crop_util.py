import numpy as np
import time
import cv2
import torch
import torch.nn.functional as F
from skimage import transform as trans
from skimage import transform
    
def get_tensor_size(mouth_region_size):
    radius = mouth_region_size//2
    radius_1_4 = radius//4
    img_h = radius * 3 + radius_1_4
    img_w = radius * 2 + radius_1_4 * 2
    return (img_w, img_h)


def get_mouth_mask_crop_rect(mouth_region_size):
    radius = mouth_region_size//2
    radius_1_4 = radius//4

    mouth_region_rect = [radius_1_4,
                        radius_1_4 + mouth_region_size,
                        radius + radius_1_4 * 2,
                        radius + mouth_region_size,
                         ]
    
    return mouth_region_rect

def get_mask_face(face, mouth_region_size):
    mouth_region_rect = get_mouth_mask_crop_rect(mouth_region_size)
    mask = face.clone()
    mask[:, :, mouth_region_rect[2]:mouth_region_rect[3], mouth_region_rect[0]:mouth_region_rect[1]] = 0
    return mask


def get_syncnet_mouth_rect(landmark):
    mouth_center = (landmark[57, :] + landmark[51, :]) / 2
    mouth_points = landmark[48:68, :]
    mouth_box = [np.min(mouth_points[:, 0]), np.max(mouth_points[:, 0]), np.min(mouth_points[:, 1]), np.max(mouth_points[:, 1])]
    h = max(mouth_box[1] - mouth_box[0], mouth_box[3] - mouth_box[2]) * 1.2
    w = 2 * h
    mouth_rect = [int(mouth_center[0] - w / 2), int(mouth_center[0] + w / 2), int(mouth_center[1] - h / 2), int(mouth_center[1] + h / 2)]

    return mouth_rect

#使用rect对图片进行截图操作，rect超出部分填充黑色
def crop_rect_with_pading(frame, rect):
    radius_clip_rect = [int(item) for item in rect]
    top = max(0, radius_clip_rect[2])
    bottom = min(radius_clip_rect[3], frame.shape[0])
    left = max(0, radius_clip_rect[0])
    right = min(radius_clip_rect[1], frame.shape[1])

    # 对截图进行padding
    img_crop = np.pad(frame[top:bottom, left:right, :],
                    ((max(0, -radius_clip_rect[2]), max(0, radius_clip_rect[3] - frame.shape[0])), 
                        (max(0, -radius_clip_rect[0]), max(0, radius_clip_rect[1] - frame.shape[1])), 
                        (0, 0)),
                    mode='constant')
    return img_crop

#使用防射矩阵对图片进行截图
def crop_face_with_tform(frame, tform, width, height):
    start_time = time.time()
    # image_transformed = transform.warp(frame, tform.inverse, output_shape=(height, width), preserve_range=True)
    
    M = np.array(tform.params)
    image_transformed = cv2.warpPerspective(frame, M, (width, height))
    
    # from scipy.ndimage import map_coordinates
    # # stack of 10 images
    # x, y = np.arange(width).astype(np.float32), np.arange(height).astype(np.float32)
    # xx, yy=np.meshgrid(x, y)

    # points = np.stack((xx, yy), axis=-1)
    # points = points.reshape(-1, 2)
    # # dummy function transforms source points into destination points

    # transformed_points = tform.inverse(points).astype(np.float32).reshape(height, width, 2)

    # # image_transformed = cv2.remap(frame, transformed_points[:, :, 0], transformed_points[:, :, 1], interpolation=cv2.INTER_LINEAR)

    # # # print("face frame cost:", time.time() - start_time)

    
    # image_transformed = None
    return image_transformed


#使用防射矩阵对图片进行截图
def restore_face_with_tform(frame, face, tform):
    from skimage import transform as trans
    from skimage import transform
    start_time = time.time()

    face_width = face.shape[1]
    face_height = face.shape[0]

    #人脸四个顶点坐标
    offset = 0
    rectangle_points = [
        [offset, offset],
        [face_width + offset, offset],
        [offset, face_height + offset],
        [face_width + offset, face_height + offset]
    ]
    #人脸四个顶点在原图中的坐标
    face_in_dest = tform.inverse(rectangle_points).astype(np.float32)

    face_in_dest_rect = [
        max(0, int(face_in_dest[:, 0].min())), min(int(face_in_dest[:, 0].max()), frame.shape[1]),
        max(0, int(face_in_dest[:, 1].min())), min(int(face_in_dest[:, 1].max()), frame.shape[0]),
    ]

    x = np.arange(face_in_dest_rect[0], face_in_dest_rect[1]).astype(np.float32)
    y = np.arange(face_in_dest_rect[2], face_in_dest_rect[3]).astype(np.float32)
    xx, yy=np.meshgrid(x, y)

    points = np.stack((xx, yy), axis=-1)
    points = points.reshape(-1, 2)
    # dummy function transforms source points into destination points

    transformed_points = tform(points).astype(np.float32).reshape(xx.shape[0], xx.shape[1], 2)

    # print("tform point cost:", time.time() - start_time)

    image_transformed = cv2.remap(face, transformed_points[:, :, 0], transformed_points[:, :, 1], interpolation=cv2.INTER_LINEAR)

    white_for_faceimg = np.ones_like(face)
    mask = cv2.remap(white_for_faceimg, transformed_points[:, :, 0], transformed_points[:, :, 1], interpolation=cv2.INTER_LINEAR)


    frame[face_in_dest_rect[2]:face_in_dest_rect[3], face_in_dest_rect[0]:face_in_dest_rect[1], :] = image_transformed * mask + frame[face_in_dest_rect[2]:face_in_dest_rect[3], face_in_dest_rect[0]:face_in_dest_rect[1], :] * (1 - mask)

    # print("restore_face_with_tform frame cost:", time.time() - start_time)

    return image_transformed


#使用防射矩阵对图片进行截图
def restore_face_with_tform_pytorch(frame, face, tform, border_mask):
    start_time = time.time()

    face_width = face.shape[2]
    face_height = face.shape[1]

    # mask = torch.ones((1, face_height, face_width), device="cuda")

    #人脸四个顶点坐标
    offset = 0
    rectangle_points = [
        [offset, offset],
        [face_width + offset, offset],
        [offset, face_height + offset],
        [face_width + offset, face_height + offset]
    ]
    #人脸四个顶点在原图中的坐标
    face_in_dest = tform.inverse(rectangle_points).astype(np.float32)

    face_in_dest_rect = [
        max(0, int(face_in_dest[:, 0].min())), min(int(face_in_dest[:, 0].max()), frame.shape[1]),
        max(0, int(face_in_dest[:, 1].min())), min(int(face_in_dest[:, 1].max()), frame.shape[0]),
    ]

    x = torch.arange(face_in_dest_rect[0], face_in_dest_rect[1], device="cuda")
    y = torch.arange(face_in_dest_rect[2], face_in_dest_rect[3], device="cuda")
    yy,xx = torch.meshgrid(y, x)
    points = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2), torch.ones((xx.shape[0], xx.shape[1], 1), device="cuda")], dim=2)
    points_homogeneous = points.view(-1, 3).unsqueeze(2)

    matrix = torch.FloatTensor(tform.params).cuda()
    transformed_points = torch.bmm(matrix.unsqueeze(0).expand(points_homogeneous.size(0), -1, -1), points_homogeneous)
    transformed_points = transformed_points.squeeze(2)[:, :2].view(xx.shape[0], xx.shape[1], 2)

    transformed_points[:, :, 0] = (transformed_points[:, :, 0] / face_width - 0.5) * 2
    transformed_points[:, :, 1] = (transformed_points[:, :, 1] / face_height - 0.5) * 2

    output_tensor = F.grid_sample(face.unsqueeze(0), transformed_points.unsqueeze(0), padding_mode='zeros', align_corners=False).squeeze(0)

    if border_mask is not None:
        mask = F.grid_sample(border_mask.unsqueeze(0), transformed_points.unsqueeze(0), padding_mode='zeros', align_corners=False).squeeze(0)
        mask = mask.permute(1, 2, 0).cpu().numpy()
    else:
        mask = None

    # print("restore_face_with_tform frame cost:", time.time() - start_time)
    output_tensor = output_tensor.permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite(f"/data/home/west/digithuman/_out/img_test/output_tensor.jpg", output_tensor)

    if mask is not None:
        blend = output_tensor * mask + frame[face_in_dest_rect[2]:face_in_dest_rect[3], face_in_dest_rect[0]:face_in_dest_rect[1], :] * (1 - mask)
    else:
        blend = output_tensor
    # cv2.imwrite(f"/data/home/west/digithuman/_out/img_test/blend.jpg", blend)

    frame[face_in_dest_rect[2]:face_in_dest_rect[3], face_in_dest_rect[0]:face_in_dest_rect[1], :] = blend

    # print("restore_face_with_tform_pytorch frame cost:", time.time() - start_time)

    return frame


# #取脸部图片垂直方向 3/4, 水平方向中点，宽度为人脸框宽度的1/2， 高度为宽度的一半
# def get_syncnet_mouth_rect(face_rect):
#     mouth_center = [(face_rect[0] + face_rect[1]) / 2, 
#                     face_rect[2] + (face_rect[3] - face_rect[2]) * 3 / 5]
#     mouth_rect_size = (face_rect[1] - face_rect[0]) / 2
#     mouth_rect = [mouth_center[0] - mouth_rect_size / 2, mouth_center[0] + mouth_rect_size / 2, 
#                     mouth_center[1] - mouth_rect_size / 4, mouth_center[1] + mouth_rect_size / 4]

#     return mouth_rect