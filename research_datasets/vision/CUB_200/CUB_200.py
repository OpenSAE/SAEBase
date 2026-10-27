"""
CUB-200-2011 fine-grained classification dataset.

Expected layout (under ``data_path``):

    <data_path>/CUB_200_2011/
        classes.txt
        images.txt
        image_class_labels.txt
        train_test_split.txt
        bounding_boxes.txt
        images/<class_name>/*.jpg
"""

import os
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image
from torch.utils.data import Dataset


# CUB-200-2011：部位 part_id → ``has_<part>_color`` 属性前缀（左右对称部位共享前缀）。
PART_ID_TO_ATTR_PREFIX: Dict[int, str] = {
    1: "has_back_color",
    2: "has_bill_color",
    3: "has_belly_color",
    4: "has_breast_color",
    5: "has_crown_color",
    6: "has_forehead_color",
    7: "has_eye_color",
    8: "has_leg_color",
    9: "has_wing_color",
    10: "has_nape_color",
    11: "has_eye_color",
    12: "has_leg_color",
    13: "has_wing_color",
    14: "has_tail_color",
    15: "has_throat_color",
}

# 属性名 ``has_*_color::<color>`` 中颜色 token → RGB（与 SSI part 面板约定一致）。
COLOR_NAME_TO_RGB: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 180, 0),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
    "rufous": (170, 85, 0),
    "grey": (150, 150, 150),
    "black": (30, 30, 30),
    "white": (240, 240, 240),
    "buff": (222, 198, 135),
    "olive": (128, 128, 0),
    "iridescent": (180, 120, 255),
}

# 与 ``datasets/CUB_200/datamodule_setup.py::_cub_transforms`` 中 eval 路径一致：
# ``Resize((512, 512))`` 后 ``CenterCrop(image_size)``（当前配置 ``image_size=448`` 时即 CenterCrop(448)）。
_CUB_GEOMETRY_RESIZE_SIDE = 512


def _map_bbox_to_input_space(
    bbox_orig: Tuple[float, float, float, float],
    *,
    H_orig: int,
    W_orig: int,
    image_size: int,
) -> Tuple[int, int, int, int]:
    """将原图像素 bbox 映射到与 ``Resize((512,512))+CenterCrop(image_size)`` 一致的输入空间。

    返回 ``(x, y, w, h)``，坐标在 ``[0, image_size]`` 内裁剪后的整数像素框。
    """
    if H_orig <= 0 or W_orig <= 0 or image_size <= 0:
        return (0, 0, max(0, int(image_size)), max(0, int(image_size)))
    r = int(_CUB_GEOMETRY_RESIZE_SIDE)
    if image_size > r:
        # 与当前 datamodule 中固定 512 再裁的假设不符；退化为整图缩放占位。
        sx = float(image_size) / float(W_orig)
        sy = float(image_size) / float(H_orig)
        x, y, w, h = (float(v) for v in bbox_orig)
        return (
            int(round(x * sx)),
            int(round(y * sy)),
            int(round(w * sx)),
            int(round(h * sy)),
        )
    sx = float(r) / float(W_orig)
    sy = float(r) / float(H_orig)
    crop_x0 = (r - image_size) // 2
    crop_y0 = (r - image_size) // 2
    x, y, w, h = (float(v) for v in bbox_orig)
    nx = x * sx - float(crop_x0)
    ny = y * sy - float(crop_y0)
    nw = w * sx
    nh = h * sy
    x0 = max(0.0, nx)
    y0 = max(0.0, ny)
    x1 = min(float(image_size), nx + nw)
    y1 = min(float(image_size), ny + nh)
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    return (int(round(x0)), int(round(y0)), int(round(bw)), int(round(bh)))


def _map_point_to_input_space(
    x_orig: float,
    y_orig: float,
    *,
    H_orig: int,
    W_orig: int,
    image_size: int,
) -> Optional[Tuple[int, int]]:
    """将原图像素坐标映射到 ``image_size``×``image_size`` 输入空间（与 bbox 同几何）。

    与 ``Resize((512,512))+CenterCrop(image_size)`` 一致。映射后严格落在 ``[0, image_size)`` 内才返回整数像素；
    否则 ``None``（落在裁切外）。
    """
    if H_orig <= 0 or W_orig <= 0 or image_size <= 0:
        return None
    r = int(_CUB_GEOMETRY_RESIZE_SIDE)
    if image_size > r:
        sx = float(image_size) / float(W_orig)
        sy = float(image_size) / float(H_orig)
        ix = int(round(float(x_orig) * sx))
        iy = int(round(float(y_orig) * sy))
        if ix < 0 or ix >= int(image_size) or iy < 0 or iy >= int(image_size):
            return None
        return (ix, iy)
    sx = float(r) / float(W_orig)
    sy = float(r) / float(H_orig)
    crop_x0 = (r - image_size) // 2
    crop_y0 = (r - image_size) // 2
    nx = float(x_orig) * sx - float(crop_x0)
    ny = float(y_orig) * sy - float(crop_y0)
    ix = int(round(nx))
    iy = int(round(ny))
    if ix < 0 or ix >= int(image_size) or iy < 0 or iy >= int(image_size):
        return None
    return (ix, iy)


class CUB200(Dataset):
    """CUB_200_2011: 200 bird species, official train/test split."""

    def __init__(
        self,
        data_path: str,
        train: bool,
        transforms,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.image_size = None if image_size is None else int(image_size)
        self.data_root = data_path
        self.cub_root = os.path.join(self.data_root, "CUB_200_2011")

        self.classes_file = os.path.join(self.cub_root, "classes.txt")
        self.image_class_labels_file = os.path.join(self.cub_root, "image_class_labels.txt")
        self.images_file = os.path.join(self.cub_root, "images.txt")
        self.train_test_split_file = os.path.join(self.cub_root, "train_test_split.txt")
        self.bounding_boxes_file = os.path.join(self.cub_root, "bounding_boxes.txt")
        self.parts_locs_file = os.path.join(self.cub_root, "parts", "part_locs.txt")
        self.attributes_file = os.path.join(self.cub_root, "attributes", "attributes.txt")
        self.image_attributes_file = os.path.join(self.cub_root, "attributes", "image_attribute_labels.txt")

        self._train_ids: List[str] = []
        self._test_ids: List[str] = []
        self._image_id_label = {}
        self._image_id_bbox: dict = {}
        self._image_id_parts: Dict[str, List[Tuple[int, float, float, int]]] = {}
        self._attr_id_to_name: Dict[int, str] = {}
        self._image_id_attrs: Dict[str, Set[int]] = {}
        self._train_path_label: List[Tuple[str, str]] = []
        self._test_path_label: List[Tuple[str, str]] = []
        self._train_bboxes: List[Optional[Tuple[float, float, float, float]]] = []
        self._test_bboxes: List[Optional[Tuple[float, float, float, float]]] = []
        self._train_parts: List[List[Tuple[int, float, float, int]]] = []
        self._test_parts: List[List[Tuple[int, float, float, int]]] = []
        self._train_attr_sets: List[Set[int]] = []
        self._test_attr_sets: List[Set[int]] = []

        self._train_test_split()
        self._get_id_to_label()
        self._get_id_to_bbox()
        self._get_id_to_parts()
        self._get_attributes()
        self._get_image_attributes()
        self._get_path_label()

    def _train_test_split(self) -> None:
        with open(self.train_test_split_file, encoding="utf-8") as f:
            for line in f:
                image_id, label = line.strip().split()
                if label == "1":
                    self._train_ids.append(image_id)
                elif label == "0":
                    self._test_ids.append(image_id)
                else:
                    raise ValueError(f"Unrecognized train_test_split flag: {label}")

    def _get_id_to_label(self) -> None:
        with open(self.image_class_labels_file, encoding="utf-8") as f:
            for line in f:
                image_id, class_id = line.strip().split()
                self._image_id_label[image_id] = class_id

    def _get_id_to_bbox(self) -> None:
        if not os.path.isfile(self.bounding_boxes_file):
            return
        with open(self.bounding_boxes_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                image_id, x, y, w, h = parts
                self._image_id_bbox[image_id] = (float(x), float(y), float(w), float(h))

    def _get_id_to_parts(self) -> None:
        if not os.path.isfile(self.parts_locs_file):
            return
        with open(self.parts_locs_file, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) != 5:
                    continue
                image_id, part_id_s, xs, ys, vis_s = tokens
                try:
                    part_id = int(part_id_s)
                    xv = float(xs)
                    yv = float(ys)
                    visible = int(vis_s)
                except (TypeError, ValueError):
                    continue
                self._image_id_parts.setdefault(image_id, []).append((part_id, xv, yv, visible))

    def _get_attributes(self) -> None:
        """加载 ``CUB_200_2011/attributes/attributes.txt``；若不存在或未能解析出任何属性名，回退到 ``data_root/attributes.txt``。"""
        primary = self.attributes_file
        fallback = os.path.join(self.data_root, "attributes.txt")

        def _read_attr_file(path: str) -> None:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split(maxsplit=1)
                    if len(tokens) != 2:
                        continue
                    try:
                        aid = int(tokens[0])
                    except ValueError:
                        continue
                    self._attr_id_to_name[aid] = tokens[1].strip()

        if os.path.isfile(primary):
            _read_attr_file(primary)
        if len(self._attr_id_to_name) > 0:
            return
        self._attr_id_to_name.clear()
        if os.path.isfile(fallback):
            _read_attr_file(fallback)

    def _get_image_attributes(self) -> None:
        if not os.path.isfile(self.image_attributes_file):
            return
        with open(self.image_attributes_file, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 3:
                    continue
                image_id = tokens[0]
                try:
                    attr_id = int(tokens[1])
                    is_present = int(tokens[2])
                except ValueError:
                    continue
                if is_present == 1:
                    self._image_id_attrs.setdefault(image_id, set()).add(attr_id)

    def _get_path_label(self) -> None:
        with open(self.images_file, encoding="utf-8") as f:
            for line in f:
                image_id, image_name = line.strip().split()
                label = self._image_id_label[image_id]
                bbox = self._image_id_bbox.get(image_id)
                parts = self._image_id_parts.get(image_id, [])
                attrs = self._image_id_attrs.get(image_id, set())
                if image_id in self._train_ids:
                    self._train_path_label.append((image_name, label))
                    self._train_bboxes.append(bbox)
                    self._train_parts.append(list(parts))
                    self._train_attr_sets.append(set(attrs))
                else:
                    self._test_path_label.append((image_name, label))
                    self._test_bboxes.append(bbox)
                    self._test_parts.append(list(parts))
                    self._test_attr_sets.append(set(attrs))

    def __len__(self) -> int:
        if self.train:
            return len(self._train_path_label)
        return len(self._test_path_label)

    def __getitem__(self, index: int):
        if self.train:
            image_name, label = self._train_path_label[index]
        else:
            image_name, label = self._test_path_label[index]
        image_path = os.path.join(self.cub_root, "images", image_name)
        img = Image.open(image_path)
        if img.mode == "L":
            img = img.convert("RGB")
        img = self.transforms(img)
        label = int(label) - 1
        return img, label

    def get_bbox_by_local_index(
        self,
        index: int,
        *,
        image_size: Optional[int] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return bbox in ``(image_size, image_size)`` input space, or ``None`` if missing.

        与 ``datamodule_setup._cub_transforms`` 的 eval 路径一致：``Resize((512,512)) + CenterCrop(image_size)``。
        ``image_size`` 缺省为 ``self.image_size``；若二者皆缺则回退 ``224``。
        """
        if self.train:
            path_label = self._train_path_label
            bboxes = self._train_bboxes
        else:
            path_label = self._test_path_label
            bboxes = self._test_bboxes
        if index < 0 or index >= len(path_label):
            raise IndexError(f"CUB local index {index} out of range (len={len(path_label)})")
        bbox_orig = bboxes[index]
        if bbox_orig is None:
            return None
        image_name, _ = path_label[index]
        image_path = os.path.join(self.cub_root, "images", image_name)
        with Image.open(image_path) as im:
            W_orig, H_orig = im.size
        sz = int(
            image_size
            if image_size is not None
            else (self.image_size if self.image_size is not None else 224)
        )
        return _map_bbox_to_input_space(bbox_orig, H_orig=H_orig, W_orig=W_orig, image_size=sz)

    def get_colored_parts_by_local_index(
        self,
        index: int,
        *,
        image_size: Optional[int] = None,
    ) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """返回 ``[(x, y, (r, g, b)), ...]``：输入空间内、可见、且存在可映射颜色属性的 part 点。

        同一 part 多个 ``is_present==1`` 的颜色属性时，取 ``attribute_id`` 升序第一个可识别颜色名。
        """
        if self.train:
            parts_list = self._train_parts
            attr_sets = self._train_attr_sets
            path_label = self._train_path_label
        else:
            parts_list = self._test_parts
            attr_sets = self._test_attr_sets
            path_label = self._test_path_label
        if index < 0 or index >= len(path_label):
            raise IndexError(f"CUB local index {index} out of range (len={len(path_label)})")
        raw_parts = parts_list[index]
        attr_set = attr_sets[index]
        image_name, _ = path_label[index]
        image_path = os.path.join(self.cub_root, "images", image_name)
        with Image.open(image_path) as im:
            W_orig, H_orig = im.size
        sz = int(
            image_size
            if image_size is not None
            else (self.image_size if self.image_size is not None else 224)
        )
        result: List[Tuple[int, int, Tuple[int, int, int]]] = []
        for part_id, x, y, visible in raw_parts:
            if int(visible) != 1:
                continue
            pt = _map_point_to_input_space(x, y, H_orig=H_orig, W_orig=W_orig, image_size=sz)
            if pt is None:
                continue
            prefix = PART_ID_TO_ATTR_PREFIX.get(int(part_id))
            if prefix is None:
                continue
            needle = prefix + "::"
            color_rgb: Optional[Tuple[int, int, int]] = None
            for attr_id in sorted(attr_set):
                attr_name = self._attr_id_to_name.get(int(attr_id), "")
                if not attr_name.startswith(needle):
                    continue
                rest = attr_name[len(needle) :].strip().lower()
                rgb = COLOR_NAME_TO_RGB.get(rest)
                if rgb is not None:
                    color_rgb = rgb
                    break
            if color_rgb is None:
                continue
            result.append((pt[0], pt[1], color_rgb))
        return result
