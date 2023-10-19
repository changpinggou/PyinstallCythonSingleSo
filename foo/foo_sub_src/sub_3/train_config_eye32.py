import os
from easydict import EasyDict as edict
# import setproctitle
# ###修改进行名
# setproctitle.setproctitle("run_landmark*_*mem")
config = edict()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.process_num = 1
config.TRAIN.prefetch_size = 20
############

config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 64
config.TRAIN.log_interval = 10  ##10 iters for a log msg
config.TRAIN.st_epoch = 0   # start epoch num
config.TRAIN.epoch = 125

config.TRAIN.lr_value_every_epoch = [0.00001, 0.0001, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]  ####lr policy
config.TRAIN.lr_decay_every_epoch = [1, 2, 50, 75, 100, 125]
config.TRAIN.weight_decay_factor = 5.e-4  ####l2
config.TRAIN.vis = False  #### if to check the training data
config.TRAIN.mix_precision = False  ##use mix precision to speedup, tf1.14 at least
config.TRAIN.opt = 'Adam'  ##Adam or SGD

config.MODEL = edict()
config.MODEL.model_path = './model_eye64_32/'  ## save directory
config.MODEL.hin = 64  # input size during training , 128,160,   depends on
config.MODEL.win = 64
config.MODEL.out_channel = [1,
                            64]  # output vector    150 points , 3 headpose  #,4 cls params,(left eye, right eye, mouth, big mouth open)
config.MODEL.landmark_num = 32

#### 'ShuffleNetV2_1.0' 'ShuffleNetV2_0.5' or MobileNetv2,
config.MODEL.net_structure = 'eye_model'
config.MODEL.pretrained_model = '/home/ji/facelandmark/model_eye64_32/epoch_0097_ION_0.031537.h5'#None
config.MODEL.export_last_ckpt = False # train model must be False

config.DATA = edict()

config.DATA.version = '232'#'232'#or'150'
config.DATA.visible = False
config.DATA.scale=1/3 #cut  3/10   transform 1/3
config.DATA.rotate=10/57
config.DATA.scale_ratioh=[0.9,1.1]
config.DATA.scale_ratiow=[0.97,1.03]
config.DATA.aug_ratio = 3
config.DATA.random_crop=[0.03,0.08]


config.DATA.train_img_path = ['/data/public/trans232_dataset/trans/all_data_1/leye/images','/data/public/trans232_dataset/trans/all_data_1/reye/images',
                              '/data/public/trans232_dataset/trans/all_data_2/leye/images','/data/public/trans232_dataset/trans/all_data_2/reye/images',

                                '/data/public/trans232_dataset/trans/325data/leye/images','/data/public/trans232_dataset/trans/325data/reye/images',
                                '/data/public/trans232_dataset/trans/427data/leye/images','/data/public/trans232_dataset/trans/427data/reye/images',
                              '/data/public/trans232_dataset/trans/325data/leye/images',
                              '/data/public/trans232_dataset/trans/325data/reye/images',
                              '/data/public/trans232_dataset/trans/427data/leye/images',
                              '/data/public/trans232_dataset/trans/427data/reye/images',
                              '/data/public/trans232_dataset/trans/325data/leye/images',
                              '/data/public/trans232_dataset/trans/325data/reye/images',
                              '/data/public/trans232_dataset/trans/427data/leye/images',
                              '/data/public/trans232_dataset/trans/427data/reye/images',
                              ]



config.DATA.train_label_path = ['/data/public/trans232_dataset/trans/all_data_1/leye/train','/data/public/trans232_dataset/trans/all_data_1/reye/train',
                                '/data/public/trans232_dataset/trans/all_data_2/leye/labels','/data/public/trans232_dataset/trans/all_data_2/reye/labels',
                                '/data/public/trans232_dataset/trans/325data/leye/labels','/data/public/trans232_dataset/trans/325data/reye/labels',
                                '/data/public/trans232_dataset/trans/427data/leye/labels','/data/public/trans232_dataset/trans/427data/reye/labels',
                                '/data/public/trans232_dataset/trans/325data/leye/labels',
                                '/data/public/trans232_dataset/trans/325data/reye/labels',
                                '/data/public/trans232_dataset/trans/427data/leye/labels',
                                '/data/public/trans232_dataset/trans/427data/reye/labels',
                                '/data/public/trans232_dataset/trans/325data/leye/labels',
                                '/data/public/trans232_dataset/trans/325data/reye/labels',
                                '/data/public/trans232_dataset/trans/427data/leye/labels',
                                '/data/public/trans232_dataset/trans/427data/reye/labels',
                                ]



config.DATA.train_txt_path = 'train.json'
config.DATA.val_img_path= ['/data/public/trans232_dataset/trans/all_data_1/leye/images','/data/public/trans232_dataset/trans/all_data_1/reye/images']
config.DATA.val_label_path =['/data/public/trans232_dataset/trans/all_data_1/leye/val','/data/public/trans232_dataset/trans/all_data_1/reye/val']

config.DATA.train_cache_path ='/home/ji/facelandmark/cache/cache_eye/cache_trian.tf-data'
config.DATA.val_cache_path ='/home/ji/facelandmark/cache/cache_eye/cache_val.tf-data'


############the model is trained with RGB mode
config.DATA.PIXEL_MEAN = [127., 127., 127.]  ###rgb
config.DATA.PIXEL_STD = [127., 127., 127.]

config.DATA.base_extend_range = [0.75, 0.75]  ###extand
config.DATA.scale_factor = [0.7, 1.35]  ###scales


# ###########################150 landmark###########################
# #轮廓
# #鼻子
# #眼睛
# #眉毛
# #嘴唇
# config.DATA.symmetry = [(0, 12),(1,11), (72, 83), (73, 82), (2, 10), (74, 81), (3, 9), (75, 80), (4, 8), (76, 79), (5, 7),
#                         (77, 78),
#
#                         (111, 114), (112, 113), (50, 53), (51, 52), (49, 54), (48, 55), (47, 56),
#
#                         (17, 30), (91, 98), (16, 31), (90, 99), (15, 32), (89, 100), (14, 33),(88,101),(13,34),
#                         (95,102),(20,35),(94,103),(19,36),(93,104),(18,37),(92,105),(97,106),(21,38),(96,107),
#
#                         (26, 39), (85, 86),(25,40),(24,41),(23,42),(84,87),(22,43),(29,44),(28,45),(27,46),
#
#                         (58, 62), (118, 125), (119, 124), (59, 61), (120, 123), (121, 122),(116,117),(134,141),(135,140),
#                         (66,68),(136,139),(137,138),(142,149),(143,148),(69,71),(144,147),(145,146),(126,133),(127,132),
#                         (63,65),(128,131),(129,130)]
#
# ##150 landmark
# leyes_pts_list = [13, 14, 15, 16, 17, 18, 19, 20, 21,
#                   88, 89, 90, 91, 92, 93, 94, 95, 96, 97]
# reyes_pts_list = [30, 31, 32, 33, 34, 35, 36, 37, 38,
#                   98, 99, 100, 101, 102, 103, 104, 105, 106, 107]
# lip_pts_list = [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
#                 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
#                 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
# face_bouding = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
#                 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
#
# print(len(leyes_pts_list))
# print(len(reyes_pts_list))
# print(len(lip_pts_list))
# print(len(face_bouding))
#
# def get_eyes_loss_weights(pts_num):
#     if pts_num == 150:
#         weights = []
#         for i in range(pts_num):
#             if i in leyes_pts_list:
#                 if i ==21 or i==96 or i==97 or i==15 or i==19:
#                     weights.append(5.0)
#                 else:
#                     weights.append(10)
#             elif i in reyes_pts_list:
#                 if i ==38 or i==106 or i==107 or i==32 or i==36:
#                     weights.append(5.0)
#                 else:
#                     weights.append(10.0)
#             elif i in lip_pts_list:
#                 weights.append(2.5)
#
#             else:
#                 weights.append(1.5)
#         weights_xy = [[x, x] for x in weights]
#
#     return np.array(weights_xy, dtype=np.float32).reshape([-1])
#
#
# config.DATA.weights150 = get_eyes_loss_weights(150)
#print(config.DATA.weights150)
config.DATA.symmetry=False
###########################232 landmark###########################
##额头

tophead_list=[212,213,214,215,216,217,218]
#轮廓
#鼻子
#眼睛
#眉毛
#嘴唇

if config.MODEL.landmark_num == 232:
    config.DATA.symmetry = [(0, 32),(1,31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26), (7, 25), (8, 24), (9, 23), (10, 22),
                            (11, 21),(12, 20),(13, 19),(14, 18),(15, 17), (212, 213), (214, 215), (216, 217),


                            (131, 132), (133, 134), (135, 136), (137, 138), (139, 140), (141, 147), (142, 146),(143, 145),
                            (223, 227),(224, 228),(225, 229),(226, 230),

                            (96, 126), (67, 97), (68, 98), (69, 99), (70, 100), (71, 101), (72, 102),(73,103),(74,104),
                            (75,105),(76,106),(77,107),(78,108),(79,109),(80,110),(81,111),(82,112),(83,113),(84,114),(85, 115),
                            (86, 116),(87, 117),(88, 118),(89, 119),(90, 120),(91, 121),(92, 122),(93, 123),(94, 124),(95, 125),
                            (219, 220),(221, 222),


                            (33, 50), (34, 51),(35,52),(36,53),(37,54),(38,55),(39,56),(40,57),(41,58),(42,59),(43, 60),(44, 61),(45, 62),
                            (46, 63),(47, 64),(48, 65),(49, 66),

                            (148, 164), (149, 163), (150, 162), (151, 161), (152, 160), (153, 159),(154,158),(155,157),
                            (165, 179), (166, 178), (167, 177), (168, 176), (169, 175), (170, 174),(171,173),
                            (197, 211), (198, 210), (199, 209), (200, 208), (201, 207), (202, 206),(203,205),(180, 196),
                            (195, 181), (194, 182), (193, 183), (192, 184), (191, 185), (190, 186),(189,187)]

if config.MODEL.landmark_num == 225:
    config.DATA.symmetry = [(0, 32),(1,31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26), (7, 25), (8, 24), (9, 23), (10, 22),
                            (11, 21),(12, 20),(13, 19),(14, 18),(15, 17),

                            (131, 132), (133, 134), (135, 136), (137, 138), (139, 140), (141, 147), (142, 146),(143, 145),
                            (216, 220),(217, 221),(218, 222),(219, 223),

                            (96, 126), (67, 97), (68, 98), (69, 99), (70, 100), (71, 101), (72, 102),(73,103),(74,104),
                            (75,105),(76,106),(77,107),(78,108),(79,109),(80,110),(81,111),(82,112),(83,113),(84,114),(85, 115),
                            (86, 116),(87, 117),(88, 118),(89, 119),(90, 120),(91, 121),(92, 122),(93, 123),(94, 124),(95, 125),
                            (212, 213),(214, 215),


                            (33, 50), (34, 51),(35,52),(36,53),(37,54),(38,55),(39,56),(40,57),(41,58),(42,59),(43, 60),(44, 61),(45, 62),
                            (46, 63),(47, 64),(48, 65),(49, 66),

                            (148, 164), (149, 163), (150, 162), (151, 161), (152, 160), (153, 159),(154,158),(155,157),
                            (165, 179), (166, 178), (167, 177), (168, 176), (169, 175), (170, 174),(171,173),
                            (197, 211), (198, 210), (199, 209), (200, 208), (201, 207), (202, 206),(203,205),(180, 196),
                            (195, 181), (194, 182), (193, 183), (192, 184), (191, 185), (190, 186),(189,187)]
if config.MODEL.landmark_num == 102:
    config.DATA.symmetry = [(0, 32), (1, 31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26), (7, 25), (8, 24), (9, 23),
                            (10, 22),(11, 21), (12, 20), (13, 19), (14, 18), (15, 17),

                            (33, 39),(34, 40), (35, 41), (36, 42), (37, 43), (38, 44),
                            (45, 54),(46, 55), (47, 56), (48, 57), (49, 58), (50, 59),(51,60),(52,61),(53,62),

                            (66,67),(68, 69),(70, 71), (72, 76), (73, 75),
                            (77, 83),(78, 82), (79, 81), (89, 91), (94, 92), (88, 84),(87, 85),(95,96),(97,98),(99,100)]
if config.MODEL.landmark_num == 79:
    config.DATA.symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),

                            (17, 23), (18, 24), (19, 25), (20, 26), (21, 27), (22, 28),
                            (29, 38), (30, 39), (31, 40), (32, 41), (33, 42), (34, 43), (35, 44), (36, 45), (37, 46),
                            (50, 51), (52, 53), (54, 55), (56, 60), (57, 59),
                            (61, 67), (62, 66), (63, 65), (73, 75), (78, 76), (72, 68), (71, 69)]




# ##232 landmark
# pt=[]
# for i in range(0,232):
#     pt.append(i)
# print(pt)


config.DATA.face_bouding = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                212,213,214,215,216,217,218]

config.DATA.leyebows= [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
config.DATA.reyebows= [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]

config.DATA.leyes_pts_list =[67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,219,221]
config.DATA.reyes_pts_list = [97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,220,222]

config.DATA.nose = [127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
        223,224,225,226,227,228,229,230]

config.DATA.tlip_pts_list = [148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180]

config.DATA.blip_pts_list = [164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 148,
                 180, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196]

config.DATA.lipsymmetry=[(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
                            (17, 33),(18, 32), (19, 31), (20, 30), (21, 29), (22, 28),(23, 27),(24, 26)]

############################232-95 landmark###############################

#0-102

##contour
##leftbrow
##rightbrow
##lefteye
##righteye
##nose
##mouth
##head

config.DATA.landmark102_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                 33,36,39,43,46,49,
                 50,53,56,60,63,66,
                 67,70,73,76,79,82,85,88,92,
                 97,100,103,106,109,112,115,118,122,
                 128,129,130,131,132,135,136,139,140,141,142,144,146,147,
                 148,152,154,156,158,160,164,167,170,172,174,177,184,188,192,200,204,208,
                 212,213,214,215,216,217,218]


config.DATA.landmark79_list = [0, 2, 5,  7,  9,  11,  13, 15, 16, 17, 19,  21,  23,  25,  27,  30,  32,
                 33,36,39,43,46,49,
                 50,53,56,60,63,66,
                 67,70,73,76,79,82,85,88,92,
                 97,100,103,106,109,112,115,118,122,
                 128,129,130,131,132,135,136,139,140,141,142,144,146,147,
                 148,152,154,156,158,160,164,167,170,172,174,177,184,188,192,200,204,208]

weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,  #####bouding
           1., 1., 1., 1., 1., 1., 1., 1., 1.,  #####nose
           1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  #####eyebows
           2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,  ####eyes
           1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5  #####mouth
           ]

weights_xy = [[x, x] for x in weights]

# config.DATA.weights = np.array(weights_xy, dtype=np.float32).reshape([-1])

config.MODEL.pruning = False  ## pruning flag  add l1 reg to bn/beta, no use for tmp
config.MODEL.pruning_bn_reg = 0.00005



config.TRACE= edict()
config.TRACE.ema_or_one_euro='euro'                 ### post process
config.TRACE.pixel_thres=1
config.TRACE.smooth_box=0.3                         ## if use euro, this will be disable
config.TRACE.smooth_landmark=0.95                   ## if use euro, this will be disable
config.TRACE.iou_thres=0.5








