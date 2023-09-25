# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('Void', 'Floor', 'Wall', 'Door', 'Ceiling', 'Decor', 'Lighting', 'Furniture', 'Construction', 'Miscellaneous', 'Plumbing', 'Storage', 'Customized', 'Cabinet0', 'Cabinet1', 'Cabinet2', 'Cabinet3', 'Table0', 'Table1', 'Table2', 'Table3', 'Sofa0', 'Sofa1', 'Sofa2', 'Sofa3', 'Electronics0', 'Electronics1', 'Electronics2', 'Electronics3', 'Chair0', 'Chair1', 'Chair2', 'Chair3', 'ArmChair0', 'ArmChair1', 'ArmChair2', 'ArmChair3', 'Bed0', 'Bed1', 'Bed2', 'Bed3', 'KidsBed0', 'KidsBed1', 'KidsBed2', 'KidsBed3', 'Stool0', 'Stool1', 'Stool2', 'Stool3', 'Platform0', 'Platform1', 'Platform2', 'Platform3', 'Sideboard0', 'Sideboard1', 'Sideboard2', 'Sideboard3', 'Bathroom0', 'Bathroom1', 'Bathroom2', 'Bathroom3', 'Window0', 'Window1', 'Window2', 'Window3', 'Appliance0', 'Appliance1', 'Appliance2', 'Appliance3', 'DiningTable0', 'DiningTable1', 'DiningTable2', 'DiningTable3', 'DeskCabinet0', 'DeskCabinet1', 'DeskCabinet2', 'DeskCabinet3', 'SingleBed0', 'SingleBed1', 'SingleBed2', 'SingleBed3', 'ClassicChair0', 'ClassicChair1', 'ClassicChair2', 'ClassicChair3',
         'CornerSideTable0', 'CornerSideTable1', 'CornerSideTable2', 'CornerSideTable3', 'Shelf0', 'Shelf1', 'Shelf2', 'Shelf3', 'Nightstand0', 'Nightstand1', 'Nightstand2', 'Nightstand3', 'ComputerChair0', 'ComputerChair1', 'ComputerChair2', 'ComputerChair3', 'DressingTable0', 'DressingTable1', 'DressingTable2', 'DressingTable3', 'Desk0', 'Desk1', 'Desk2', 'Desk3', 'DressingChair0', 'DressingChair1', 'DressingChair2', 'DressingChair3', 'Wardrobe0', 'Wardrobe1', 'Wardrobe2', 'Wardrobe3', 'BunkBed0', 'BunkBed1', 'BunkBed2', 'BunkBed3', 'BookcaseCabinet0', 'BookcaseCabinet1', 'BookcaseCabinet2', 'BookcaseCabinet3', 'CafeChair0', 'CafeChair1', 'CafeChair2', 'CafeChair3', 'CoffeTable0', 'CoffeTable1', 'CoffeTable2', 'CoffeTable3', 'KingSizedBed0', 'KingSizedBed1', 'KingSizedBed2', 'KingSizedBed3', 'MultiSeatSofa0', 'MultiSeatSofa1', 'MultiSeatSofa2', 'MultiSeatSofa3', 'SideCabinet0', 'SideCabinet1', 'SideCabinet2', 'SideCabinet3', 'ShoeCabinet0', 'ShoeCabinet1', 'ShoeCabinet2', 'ShoeCabinet3', 'BedSofa0', 'BedSofa1', 'BedSofa2', 'BedSofa3'),
        # ('void', 'smartcustomizedceiling', 'cabinet/lightband', 'tea table', 'cornice', 'sewerpipe', 'children cabinet', 'hole', 'ceiling lamp', 'chaise longue sofa', 'lazy sofa', 'appliance', 'round end table', 'build element', 'dining chair', 'others', 'armchair', 'bed', 'two-seat sofa', 'lighting', 'kids bed', 'pocket', 'storage unit', 'media unit', 'slabside', 'footstool / sofastool / bed end stool / stool', '300 - on top of others', 'customizedplatform', 'sideboard / side cabinet / console', 'plants', 'ceiling', 'slabtop', 'pendant lamp', 'lightband', 'electric', 'pier/stool', 'table', 'extrusioncustomizedceilingmodel', 'baseboard', 'front', 'wallinner', 'basin', 'bath', 'customizedpersonalizedmodel', 'baywindow', 'customizedfurniture', 'sofa', 'kitchen cabinet', 'cabinet', 'walltop', 'chair', 'floor', 'customizedceiling', '500 - attach to ceiling', 'customizedbackgroundmodel', 'drawer chest / corner cabinet', 'tv stand', '400 - attach to wall', 'window', 'art', 'back', 'accessory', '200 - on the floor', 'beam', 'stair', 'wine cooler', 'outdoor furniture', 'double bed', 'dining table', 'cabinet/shelf/desk', 'single bed', 'classic chinese chair', 'corner/side table', 'flue', 'shelf', 'customizedfeaturewall', 'nightstand', 'recreation', 'lounge chair / book-chair / computer chair', 'slabbottom', 'dressing table', 'desk', 'column', 'dressing chair', 'wardrobe', 'extrusioncustomizedbackgroundwall', 'electronics', 'bunk bed', 'bed frame', 'three-seat / multi-person sofa', 'customizedfixedfurniture', 'bookcase / jewelry armoire', 'mirror', 'wallbottom', 'barstool', 'wallouter', 'l-shaped sofa', 'customized_wainscot', 'door', 'lounge chair / cafe chair / office chair', 'coffee table', 'king-size bed', 'three-seat / multi-seat sofa', 'sideboard / side cabinet / console table', 'loveseat sofa', 'wine cabinet', 'bar', 'shoe cabinet', 'couch bed', 'hanging chair', 'folding chair', 'u-shaped sofa', 'floor lamp', 'wall lamp', '114', '115', '116', '117', '118', '119', '120', '121', '122'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(237, 171, 161), (243, 241, 13), (83, 189, 236), (58, 14, 41), (53, 158, 221), (71, 160, 224), (115, 187, 120), (85, 96, 93), (99, 21, 43), (189, 241, 187), (245, 226, 254), (38, 69, 251), (20, 202, 33), (67, 173, 200), (233, 235, 48), (78, 42, 79), (221, 165, 156), (111, 168, 160), (146, 45, 56), (95, 156, 179), (213, 76, 245), (80, 215, 219), (143, 39, 11), (47, 151, 10), (224, 254, 197), (27, 10, 230), (96, 36, 200), (210, 19, 7), (203, 218, 143), (107, 16, 25), (252, 117, 95), (182, 60, 146), (248, 204, 54), (185, 194, 117), (0, 147, 238), (50, 153, 22), (184, 94, 169), (147, 118, 217), (3, 198, 245), (86, 79, 227), (162, 111, 17), (62, 88, 0), (6, 4, 241), (158, 24, 214), (179, 179, 203), (45, 205, 68), (231, 101, 44), (157, 109, 129), (126, 82, 158), (56, 55, 92), (215, 246, 125), (31, 128, 234), (11, 99, 136), (196, 66, 122), (188, 31, 101), (81, 15, 96), (15, 198, 251), (131, 36, 210), (193, 251, 9), (64, 146, 190), (133, 218, 41), (88, 250, 231), (153, 192, 207), (17, 53, 15), (66, 212, 183), (231, 129, 150), (168, 229, 186), (92, 248, 165), (216, 75, 114), (55, 92, 53), (25, 136, 75), (160, 196, 194), (208, 83, 90), (69, 62, 5), (128, 61, 174), (125, 169, 64), (190, 40, 148),
         (149, 244, 62), (251, 113, 31), (121, 177, 215), (75, 106, 19), (18, 233, 86), (52, 26, 195), (199, 52, 47), (145, 114, 168), (170, 230, 237), (11, 149, 3), (69, 165, 126), (48, 182, 26), (178, 134, 183), (166, 185, 83), (182, 79, 73), (77, 210, 139), (140, 228, 100), (31, 85, 225), (130, 181, 76), (119, 103, 2), (41, 13, 132), (37, 176, 129), (198, 163, 35), (239, 223, 151), (222, 122, 230), (98, 214, 176), (89, 8, 50), (74, 142, 105), (165, 132, 165), (34, 174, 106), (91, 105, 110), (6, 57, 85), (201, 209, 88), (117, 222, 137), (110, 125, 134), (112, 0, 104), (195, 98, 80), (227, 208, 111), (118, 200, 155), (175, 191, 162), (123, 237, 212), (210, 140, 242), (14, 34, 65), (44, 238, 38), (204, 67, 206), (107, 50, 210), (225, 184, 81), (102, 120, 23), (174, 25, 71), (61, 9, 222), (104, 44, 123), (138, 138, 181), (248, 144, 202), (23, 21, 189), (3, 221, 67), (253, 155, 153), (244, 143, 60), (214, 162, 29), (22, 71, 171), (135, 122, 193), (205, 248, 247), (135, 86, 133), (9, 29, 173), (34, 47, 118), (229, 89, 108), (152, 125, 51), (141, 2, 178), (154, 94, 59), (170, 48, 141), (42, 235, 145), (241, 58, 249), (236, 73, 115), (103, 31, 20), (29, 70, 38), (162, 5, 70), (173, 133, 99), (219, 110, 31)]
        # [(213, 208, 241), (109, 103, 171), (22, 159, 198), (9, 78, 48), (183, 189, 201), (178, 210, 133), (181, 28, 160), (29, 69, 41), (204, 2, 237), (6, 54, 116), (220, 41, 24), (13, 64, 97), (86, 149, 203), (69, 73, 146), (240, 161, 176), (148, 168, 178), (24, 141, 22), (113, 45, 215), (97, 114, 72), (207, 50, 91), (49, 166, 206), (126, 246, 207), (225, 88, 185), (154, 8, 28), (88, 102, 162), (156, 26, 190), (76, 249, 169), (80, 164, 55), (194, 162, 26), (173, 230, 39), (162, 91, 199), (20, 184, 131), (252, 214, 226), (248, 181, 104), (76, 134, 3), (140, 97, 63), (99, 253, 59), (176, 1, 137), (36, 202, 245), (58, 211, 165), (130, 122, 158), (250, 17, 36), (192, 200, 156), (235, 193, 19), (89, 72, 215), (146, 13, 235), (110, 229, 9), (229, 152, 11), (223, 10, 154), (138, 219, 47), (92, 154, 213), (65, 184, 61), (241, 178, 124), (123, 137, 76), (32, 134, 186), (175, 5, 167), (73, 106, 127), (144, 170, 7), (169, 216, 50), (203, 187, 18), (167, 15, 111), (190, 93, 172), (25, 172, 96), (238, 206, 93), (63, 118, 105), (207, 233, 44), (67, 140, 122), (193, 37, 220), (37, 111, 179), (218, 223, 181), (142, 53, 222), (254, 49, 68), (18, 29, 113), (95, 126, 75), (0, 101, 88), (242, 60, 254), (120, 235, 93), (82, 243, 139), (129, 240, 65), (14, 35, 73), (12, 144, 209), (159, 121, 151), (104, 237, 228), (4, 20, 225), (151, 94, 243), (98, 59, 130), (200, 247, 1), (117, 68, 231), (227, 108, 109), (230, 85, 240), (60, 79, 14), (159, 198, 30), (107, 20, 101), (34, 31, 53), (244, 150, 69), (187, 23, 119), (170, 145, 124), (45, 192, 78), (115, 241, 195), (40, 84, 114), (213, 128, 80), (163, 95, 194), (83, 35, 32), (54, 227, 183), (151, 63, 86), (53, 6, 34), (57, 47, 218), (234, 156, 250), (209, 177, 149), (29, 56, 143), (122, 224, 247), (134, 76, 192), (42, 128, 39), (184, 42, 135), (135, 251, 12), (216, 221, 57), (103, 116, 142), (2, 203, 250), (128, 130, 101), (45, 81, 5), (72, 195, 232), (198, 113, 149), (49, 176, 83)]

    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        print(self.metainfo['classes'])
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
