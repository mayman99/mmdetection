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
        ('Void', 'Floor', 'Wall', 'Door', 'Ceiling', 'Decor', 'Lighting', 'Furniture', 'Construction', 'Miscellaneous', 'Plumbing', 'Storage', 'Customized', 'Plants', 'Cabinet0', 'Cabinet1', 'Cabinet2', 'Cabinet3', 'Cabinet4', 'Cabinet5', 'Cabinet6', 'Cabinet7', 'Table0', 'Table1', 'Table2', 'Table3', 'Table4', 'Table5', 'Table6', 'Table7', 'Sofa0', 'Sofa1', 'Sofa2', 'Sofa3', 'Sofa4', 'Sofa5', 'Sofa6', 'Sofa7', 'Electronics0', 'Electronics1', 'Electronics2', 'Electronics3', 'Electronics4', 'Electronics5', 'Electronics6', 'Electronics7', 'Chair0', 'Chair1', 'Chair2', 'Chair3', 'Chair4', 'Chair5', 'Chair6', 'Chair7', 'ArmChair0', 'ArmChair1', 'ArmChair2', 'ArmChair3', 'ArmChair4', 'ArmChair5', 'ArmChair6', 'ArmChair7', 'Bed0', 'Bed1', 'Bed2', 'Bed3', 'Bed4', 'Bed5', 'Bed6', 'Bed7', 'KidsBed0', 'KidsBed1', 'KidsBed2', 'KidsBed3', 'KidsBed4', 'KidsBed5', 'KidsBed6', 'KidsBed7', 'Stool0', 'Stool1', 'Stool2', 'Stool3', 'Stool4', 'Stool5', 'Stool6', 'Stool7', 'Platform0', 'Platform1', 'Platform2', 'Platform3', 'Platform4', 'Platform5', 'Platform6', 'Platform7', 'Sideboard0', 'Sideboard1', 'Sideboard2', 'Sideboard3', 'Sideboard4', 'Sideboard5', 'Sideboard6', 'Sideboard7', 'Bathroom0', 'Bathroom1', 'Bathroom2', 'Bathroom3', 'Bathroom4', 'Bathroom5', 'Bathroom6', 'Bathroom7', 'Window0', 'Window1', 'Window2', 'Window3', 'Window4', 'Window5', 'Window6', 'Window7', 'Tvstand0', 'Tvstand1', 'Tvstand2', 'Tvstand3', 'Tvstand4', 'Tvstand5', 'Tvstand6', 'Tvstand7', 'Appliance0', 'Appliance1', 'Appliance2', 'Appliance3', 'Appliance4', 'Appliance5', 'Appliance6', 'Appliance7', 'DiningTable0', 'DiningTable1', 'DiningTable2', 'DiningTable3', 'DiningTable4', 'DiningTable5', 'DiningTable6', 'DiningTable7', 'DeskCabinet0', 'DeskCabinet1', 'DeskCabinet2', 'DeskCabinet3', 'DeskCabinet4', 'DeskCabinet5', 'DeskCabinet6', 'DeskCabinet7', 'SingleBed0', 'SingleBed1', 'SingleBed2', 'SingleBed3', 'SingleBed4', 'SingleBed5', 'SingleBed6', 'SingleBed7', 'ClassicChair0', 'ClassicChair1', 'ClassicChair2', 'ClassicChair3', 'ClassicChair4', 'ClassicChair5', 'ClassicChair6', 'ClassicChair7',
         'CornerSideTable0', 'CornerSideTable1', 'CornerSideTable2', 'CornerSideTable3', 'CornerSideTable4', 'CornerSideTable5', 'CornerSideTable6', 'CornerSideTable7', 'Shelf0', 'Shelf1', 'Shelf2', 'Shelf3', 'Shelf4', 'Shelf5', 'Shelf6', 'Shelf7', 'Nightstand0', 'Nightstand1', 'Nightstand2', 'Nightstand3', 'Nightstand4', 'Nightstand5', 'Nightstand6', 'Nightstand7', 'ComputerChair0', 'ComputerChair1', 'ComputerChair2', 'ComputerChair3', 'ComputerChair4', 'ComputerChair5', 'ComputerChair6', 'ComputerChair7', 'DressingTable0', 'DressingTable1', 'DressingTable2', 'DressingTable3', 'DressingTable4', 'DressingTable5', 'DressingTable6', 'DressingTable7', 'Desk0', 'Desk1', 'Desk2', 'Desk3', 'Desk4', 'Desk5', 'Desk6', 'Desk7', 'DressingChair0', 'DressingChair1', 'DressingChair2', 'DressingChair3', 'DressingChair4', 'DressingChair5', 'DressingChair6', 'DressingChair7', 'Wardrobe0', 'Wardrobe1', 'Wardrobe2', 'Wardrobe3', 'Wardrobe4', 'Wardrobe5', 'Wardrobe6', 'Wardrobe7', 'BunkBed0', 'BunkBed1', 'BunkBed2', 'BunkBed3', 'BunkBed4', 'BunkBed5', 'BunkBed6', 'BunkBed7', 'BookcaseCabinet0', 'BookcaseCabinet1', 'BookcaseCabinet2', 'BookcaseCabinet3', 'BookcaseCabinet4', 'BookcaseCabinet5', 'BookcaseCabinet6', 'BookcaseCabinet7', 'CafeChair0', 'CafeChair1', 'CafeChair2', 'CafeChair3', 'CafeChair4', 'CafeChair5', 'CafeChair6', 'CafeChair7', 'CoffeTable0', 'CoffeTable1', 'CoffeTable2', 'CoffeTable3', 'CoffeTable4', 'CoffeTable5', 'CoffeTable6', 'CoffeTable7', 'KingSizedBed0', 'KingSizedBed1', 'KingSizedBed2', 'KingSizedBed3', 'KingSizedBed4', 'KingSizedBed5', 'KingSizedBed6', 'KingSizedBed7', 'MultiSeatSofa0', 'MultiSeatSofa1', 'MultiSeatSofa2', 'MultiSeatSofa3', 'MultiSeatSofa4', 'MultiSeatSofa5', 'MultiSeatSofa6', 'MultiSeatSofa7', 'SideCabinet0', 'SideCabinet1', 'SideCabinet2', 'SideCabinet3', 'SideCabinet4', 'SideCabinet5', 'SideCabinet6', 'SideCabinet7', 'ShoeCabinet0', 'ShoeCabinet1', 'ShoeCabinet2', 'ShoeCabinet3', 'ShoeCabinet4', 'ShoeCabinet5', 'ShoeCabinet6', 'ShoeCabinet7', 'BedSofa0', 'BedSofa1', 'BedSofa2', 'BedSofa3', 'BedSofa4', 'BedSofa5', 'BedSofa6', 'BedSofa7'),
        # ('Void', 'Floor', 'Wall', 'Door', 'Ceiling', 'Decor', 'Lighting', 'Furniture', 'Construction', 'Miscellaneous', 'Plumbing', 'Storage', 'Customized', 'Cabinet0', 'Cabinet1', 'Cabinet2', 'Cabinet3', 'Table0', 'Table1', 'Table2', 'Table3', 'Sofa0', 'Sofa1', 'Sofa2', 'Sofa3', 'Electronics0', 'Electronics1', 'Electronics2', 'Electronics3', 'Chair0', 'Chair1', 'Chair2', 'Chair3', 'ArmChair0', 'ArmChair1', 'ArmChair2', 'ArmChair3', 'Bed0', 'Bed1', 'Bed2', 'Bed3', 'KidsBed0', 'KidsBed1', 'KidsBed2', 'KidsBed3', 'Stool0', 'Stool1', 'Stool2', 'Stool3', 'Platform0', 'Platform1', 'Platform2', 'Platform3', 'Sideboard0', 'Sideboard1', 'Sideboard2', 'Sideboard3', 'Bathroom0', 'Bathroom1', 'Bathroom2', 'Bathroom3', 'Window0', 'Window1', 'Window2', 'Window3', 'Appliance0', 'Appliance1', 'Appliance2', 'Appliance3', 'DiningTable0', 'DiningTable1', 'DiningTable2', 'DiningTable3', 'DeskCabinet0', 'DeskCabinet1', 'DeskCabinet2', 'DeskCabinet3', 'SingleBed0', 'SingleBed1', 'SingleBed2', 'SingleBed3', 'ClassicChair0', 'ClassicChair1', 'ClassicChair2', 'ClassicChair3',
        #  'CornerSideTable0', 'CornerSideTable1', 'CornerSideTable2', 'CornerSideTable3', 'Shelf0', 'Shelf1', 'Shelf2', 'Shelf3', 'Nightstand0', 'Nightstand1', 'Nightstand2', 'Nightstand3', 'ComputerChair0', 'ComputerChair1', 'ComputerChair2', 'ComputerChair3', 'DressingTable0', 'DressingTable1', 'DressingTable2', 'DressingTable3', 'Desk0', 'Desk1', 'Desk2', 'Desk3', 'DressingChair0', 'DressingChair1', 'DressingChair2', 'DressingChair3', 'Wardrobe0', 'Wardrobe1', 'Wardrobe2', 'Wardrobe3', 'BunkBed0', 'BunkBed1', 'BunkBed2', 'BunkBed3', 'BookcaseCabinet0', 'BookcaseCabinet1', 'BookcaseCabinet2', 'BookcaseCabinet3', 'CafeChair0', 'CafeChair1', 'CafeChair2', 'CafeChair3', 'CoffeTable0', 'CoffeTable1', 'CoffeTable2', 'CoffeTable3', 'KingSizedBed0', 'KingSizedBed1', 'KingSizedBed2', 'KingSizedBed3', 'MultiSeatSofa0', 'MultiSeatSofa1', 'MultiSeatSofa2', 'MultiSeatSofa3', 'SideCabinet0', 'SideCabinet1', 'SideCabinet2', 'SideCabinet3', 'ShoeCabinet0', 'ShoeCabinet1', 'ShoeCabinet2', 'ShoeCabinet3', 'BedSofa0', 'BedSofa1', 'BedSofa2', 'BedSofa3'),
        # ('void', 'smartcustomizedceiling', 'cabinet/lightband', 'tea table', 'cornice', 'sewerpipe', 'children cabinet', 'hole', 'ceiling lamp', 'chaise longue sofa', 'lazy sofa', 'appliance', 'round end table', 'build element', 'dining chair', 'others', 'armchair', 'bed', 'two-seat sofa', 'lighting', 'kids bed', 'pocket', 'storage unit', 'media unit', 'slabside', 'footstool / sofastool / bed end stool / stool', '300 - on top of others', 'customizedplatform', 'sideboard / side cabinet / console', 'plants', 'ceiling', 'slabtop', 'pendant lamp', 'lightband', 'electric', 'pier/stool', 'table', 'extrusioncustomizedceilingmodel', 'baseboard', 'front', 'wallinner', 'basin', 'bath', 'customizedpersonalizedmodel', 'baywindow', 'customizedfurniture', 'sofa', 'kitchen cabinet', 'cabinet', 'walltop', 'chair', 'floor', 'customizedceiling', '500 - attach to ceiling', 'customizedbackgroundmodel', 'drawer chest / corner cabinet', 'tv stand', '400 - attach to wall', 'window', 'art', 'back', 'accessory',
        #  '200 - on the floor', 'beam', 'stair', 'wine cooler', 'outdoor furniture', 'double bed', 'dining table', 'cabinet/shelf/desk', 'single bed', 'classic chinese chair', 'corner/side table', 'flue', 'shelf', 'customizedfeaturewall', 'nightstand', 'recreation', 'lounge chair / book-chair / computer chair', 'slabbottom', 'dressing table', 'desk', 'column', 'dressing chair', 'wardrobe', 'extrusioncustomizedbackgroundwall', 'electronics', 'bunk bed', 'bed frame', 'three-seat / multi-person sofa', 'customizedfixedfurniture', 'bookcase / jewelry armoire', 'mirror', 'wallbottom', 'barstool', 'wallouter', 'l-shaped sofa', 'customized_wainscot', 'door', 'lounge chair / cafe chair / office chair', 'coffee table', 'king-size bed', 'three-seat / multi-seat sofa', 'sideboard / side cabinet / console table', 'loveseat sofa', 'wine cabinet', 'bar', 'shoe cabinet', 'couch bed', 'hanging chair', 'folding chair', 'u-shaped sofa', 'floor lamp', 'wall lamp', '114', '115', '116', '117', '118', '119', '120', '121'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(218, 180, 18), (236, 232, 48), (97, 176, 159), (81, 190, 28), (116, 185, 122), (239, 196, 162), (194, 44, 97), (4, 22, 73), (60, 242, 242), (3, 35, 5), (16, 24, 37), (186, 159, 178), (124, 46, 66), (133, 102, 236), (139, 141, 229), (203, 105, 254), (63, 236, 75), (198, 115, 60), (163, 249, 215), (159, 174, 190), (11, 111, 247), (146, 238, 179), (183, 125, 251), (148, 128, 70), (195, 152, 239), (54, 63, 102), (200, 71, 85), (162, 17, 46), (65, 251, 241), (99, 133, 99), (127, 30, 41), (30, 122, 34), (215, 227, 74), (23, 229, 25), (26, 220, 9), (85, 142, 62), (45, 38, 87), (156, 85, 156), (55, 138, 168), (137, 143, 92), (68, 95, 12), (141, 155, 28), (75, 66, 107), (177, 22, 155), (135, 49, 20), (230, 244, 119), (44, 5, 196), (65, 217, 214), (176, 154, 176), (193, 130, 15), (25, 31, 31), (73, 160, 64), (190, 63, 38), (46, 56, 209), (34, 214, 166), (111, 6, 175), (174, 15, 135), (72, 205, 126), (150, 34, 177), (118, 222, 69), (218, 116, 197), (119, 99, 222), (103, 252, 249), (142, 193, 205), (84, 126, 126), (94, 216, 54), (107, 234, 16), (152, 167, 62), (110, 15, 40), (223, 187, 239), (76, 208, 106), (89, 78, 8), (145, 237, 53), (61, 50, 4), (231, 168, 244), (114, 234, 196), (229, 57, 44), (196, 183, 1), (15, 19, 111), (102, 10, 164), (225, 212, 231), (128, 179, 145), (175, 134, 6), (167, 9, 89), (127, 197, 173), (199, 117, 210), (137, 207, 31), (123, 61, 230), (22, 99, 226), (239, 228, 55), (206, 178, 50), (195, 184, 147), (171, 109, 57), (21, 201, 136), (241, 231, 17), (87, 225, 118), (223, 148, 157), (249, 16, 247), (108, 74, 29), (143, 37, 84), (209, 221, 24), (17, 202, 39), (81, 245, 99), (153, 135, 22), (108, 53, 192), (122, 209, 183), (97, 122, 23), (32, 254, 45), (120, 83, 195), (238, 250, 33), (250, 65, 182), (5, 50, 169), (62, 81, 111), (212, 198, 219), (125, 103, 186), (236, 224, 11), (251, 1, 205), (187, 139, 149), (79, 72, 150), (91, 76, 80), (253, 169, 1), (92, 38, 69), (7, 157, 81), (27, 9, 79), (113, 41, 134), (46, 98, 139), (38, 59, 101), (185, 163, 207), (244, 8, 232), (42, 246, 187), (66, 127, 113), (101, 206, 71), (213, 181, 139), (235, 80, 167), (226, 76, 143), (214, 2, 42), (121, 114, 161), (253, 21, 144), (34, 131, 238), (248, 40, 7), (27, 137, 116), (140, 46, 105), (171, 180, 91), (53, 111, 93), (40, 26, 136), (207, 90, 221), (52, 14, 211), (238, 217, 235), (166, 48, 110), (247, 146, 202), (77, 94, 216),
         (173, 172, 251), (88, 68, 37), (158, 164, 233), (254, 215, 206), (12, 213, 65), (39, 69, 133), (209, 211, 245), (197, 248, 142), (182, 73, 18), (3, 79, 88), (210, 83, 53), (181, 239, 217), (29, 18, 104), (201, 11, 132), (23, 48, 121), (131, 167, 83), (180, 70, 26), (33, 52, 96), (105, 87, 121), (179, 252, 82), (250, 27, 98), (169, 4, 201), (35, 253, 151), (138, 61, 77), (130, 33, 179), (188, 150, 41), (25, 243, 157), (172, 24, 3), (178, 87, 105), (149, 188, 164), (162, 136, 128), (191, 160, 154), (202, 147, 237), (136, 200, 137), (36, 60, 95), (161, 186, 86), (49, 133, 218), (133, 191, 184), (47, 86, 194), (99, 124, 109), (79, 204, 13), (192, 176, 76), (31, 101, 129), (72, 55, 225), (219, 228, 250), (216, 245, 117), (191, 29, 91), (106, 58, 140), (48, 11, 125), (232, 6, 85), (126, 165, 45), (233, 72, 213), (154, 121, 36), (90, 192, 57), (70, 204, 245), (117, 101, 8), (155, 226, 80), (56, 54, 212), (30, 32, 225), (64, 0, 52), (19, 247, 68), (105, 150, 198), (234, 162, 235), (165, 223, 174), (217, 120, 60), (157, 170, 24), (8, 144, 221), (163, 42, 49), (37, 117, 94), (18, 230, 241), (49, 89, 115), (1, 236, 181), (104, 140, 114), (166, 173, 162), (230, 155, 108), (0, 104, 131), (243, 175, 243), (77, 138, 58), (226, 145, 160), (1, 113, 2), (174, 183, 124), (95, 195, 93), (7, 107, 200), (170, 20, 188), (222, 118, 148), (144, 28, 222), (220, 85, 146), (153, 44, 0), (98, 142, 50), (221, 223, 14), (90, 156, 172), (11, 109, 10), (142, 96, 116), (122, 199, 141), (83, 91, 170), (41, 58, 43), (14, 75, 112), (245, 97, 248), (245, 40, 203), (13, 210, 185), (74, 209, 193), (227, 214, 124), (57, 194, 120), (86, 171, 142), (147, 108, 207), (129, 239, 72), (20, 82, 181), (92, 164, 227), (246, 197, 56), (93, 92, 177), (184, 130, 159), (134, 104, 228), (151, 189, 200), (9, 192, 73), (69, 153, 47), (228, 53, 169), (115, 241, 27), (112, 113, 67), (116, 79, 233), (101, 219, 172), (42, 93, 253), (187, 161, 21), (203, 1, 188), (71, 151, 220), (43, 67, 33), (216, 33, 127), (204, 88, 252), (80, 43, 191), (211, 127, 209), (240, 241, 152), (112, 131, 224), (58, 36, 13), (60, 149, 130), (19, 20, 165), (82, 26, 132), (180, 158, 193), (6, 3, 156), (168, 188, 204), (59, 170, 148), (189, 94, 19), (160, 177, 63), (57, 64, 230), (10, 119, 153), (151, 202, 100), (208, 233, 77), (52, 107, 103), (50, 123, 189), (68, 12, 32), (243, 220, 89), (205, 29, 214), (131, 68, 59)]
        # [(213, 208, 241), (109, 103, 171), (22, 159, 198), (9, 78, 48), (183, 189, 201), (178, 210, 133), (181, 28, 160), (29, 69, 41), (204, 2, 237), (6, 54, 116), (220, 41, 24), (13, 64, 97), (86, 149, 203), (69, 73, 146), (240, 161, 176), (148, 168, 178), (24, 141, 22), (113, 45, 215), (97, 114, 72), (207, 50, 91), (49, 166, 206), (126, 246, 207), (225, 88, 185), (154, 8, 28), (88, 102, 162), (156, 26, 190), (76, 249, 169), (80, 164, 55), (194, 162, 26), (173, 230, 39), (162, 91, 199), (20, 184, 131), (252, 214, 226), (248, 181, 104), (76, 134, 3), (140, 97, 63), (99, 253, 59), (176, 1, 137), (36, 202, 245), (58, 211, 165), (130, 122, 158), (250, 17, 36), (192, 200, 156), (235, 193, 19), (89, 72, 215), (146, 13, 235), (110, 229, 9), (229, 152, 11), (223, 10, 154), (138, 219, 47), (92, 154, 213), (65, 184, 61), (241, 178, 124), (123, 137, 76), (32, 134, 186), (175, 5, 167), (73, 106, 127), (144, 170, 7), (169, 216, 50), (203, 187, 18), (167, 15, 111),
        #  (190, 93, 172), (25, 172, 96), (238, 206, 93), (63, 118, 105), (207, 233, 44), (67, 140, 122), (193, 37, 220), (37, 111, 179), (218, 223, 181), (142, 53, 222), (254, 49, 68), (18, 29, 113), (95, 126, 75), (0, 101, 88), (242, 60, 254), (120, 235, 93), (82, 243, 139), (129, 240, 65), (14, 35, 73), (12, 144, 209), (159, 121, 151), (104, 237, 228), (4, 20, 225), (151, 94, 243), (98, 59, 130), (200, 247, 1), (117, 68, 231), (227, 108, 109), (230, 85, 240), (60, 79, 14), (159, 198, 30), (107, 20, 101), (34, 31, 53), (244, 150, 69), (187, 23, 119), (170, 145, 124), (45, 192, 78), (115, 241, 195), (40, 84, 114), (213, 128, 80), (163, 95, 194), (83, 35, 32), (54, 227, 183), (151, 63, 86), (53, 6, 34), (57, 47, 218), (234, 156, 250), (209, 177, 149), (29, 56, 143), (122, 224, 247), (134, 76, 192), (42, 128, 39), (184, 42, 135), (135, 251, 12), (216, 221, 57), (103, 116, 142), (2, 203, 250), (128, 130, 101), (45, 81, 5), (72, 195, 232), (198, 113, 149)]

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
        print(data_info)
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
