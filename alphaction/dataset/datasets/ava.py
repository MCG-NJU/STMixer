import os
import torch.utils.data as data
import time
import torch
import numpy as np
from alphaction.structures.bounding_box import BoxList
from collections import defaultdict
from alphaction.utils.video_decode import av_decode_video

import json

# This is used to avoid pytorch issuse #13246
class NpInfoDict(object):
    def __init__(self, info_dict, key_type=None, value_type=None):
        keys = sorted(list(info_dict.keys()))
        self.key_arr = np.array(keys, dtype=key_type)
        self.val_arr = np.array([info_dict[k] for k in keys], dtype=value_type)
        # should not be used in dataset __getitem__
        self._key_idx_map = {k:i for i,k in enumerate(keys)}
    def __getitem__(self, idx):
        return self.key_arr[idx], self.val_arr[idx]
    def __len__(self):
        return len(self.key_arr)
    def convert_key(self, org_key):
        # convert relevant variable whose original value is in the key set.
        # should not be used in __getitem__
        return self._key_idx_map[org_key]

# This is used to avoid pytorch issuse #13246
class NpBoxDict(object):
    def __init__(self, id_to_box_dict, key_list=None, value_types=[]):
        # clip2ann: {image_id: [{bbox, packed_act},...], ...}
        # clip_ids: [image_id, ...]
        # [("bbox", np.float32), ("packed_act", np.uint8)]
        value_fields, value_types = list(zip(*value_types))
        assert "bbox" in value_fields

        if key_list is None:
            key_list = sorted(list(id_to_box_dict.keys()))
        self.length = len(key_list)

        pointer_list = []
        value_lists = {field: [] for field in value_fields}
        # {bbox:[], packed_act:[]}
        cur = 0
        pointer_list.append(cur)
        for k in key_list:
            box_infos = id_to_box_dict[k]
            cur += len(box_infos)
            # num of boxes
            pointer_list.append(cur)
            for box_info in box_infos:
                for field in value_fields:
                    value_lists[field].append(box_info[field])
        self.pointer_arr = np.array(pointer_list, dtype=np.int32)
        # [0, num1, num1+num2, num1+num2+num3, ...]
        self.attr_names = np.array(["vfield_" + field for field in value_fields])
        for field_name, value_type, attr_name in zip(value_fields, value_types, self.attr_names):
            setattr(self, attr_name, np.array(value_lists[field_name], dtype=value_type))
        # vfield_bbox: num_all_boxes, 4
        # vfield_packed_act: num_all_boxes, 8

    def __getitem__(self, idx):
        l_pointer = self.pointer_arr[idx]
        r_pointer = self.pointer_arr[idx + 1]
        ret_val = [getattr(self, attr_name)[l_pointer:r_pointer] for attr_name in self.attr_names]
        return ret_val

    def __len__(self):
        return self.length

class AVAVideoDataset(data.Dataset):
    def __init__(self, video_root, ann_file, remove_clips_without_annotations, frame_span,
                 eval_file_paths={}, transforms=None):

        print('loading annotations into memory...')
        tic = time.time()
        json_dict = json.load(open(ann_file, 'r'))
        assert type(json_dict) == dict, 'annotation file format {} not supported'.format(type(json_dict))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        self.video_root = video_root
        self.transforms = transforms
        self.frame_span = frame_span

        # These two attributes are used during ava evaluation...
        # Maybe there is a better implementation
        self.eval_file_paths = eval_file_paths

        clip2ann = defaultdict(list)
        if "annotations" in json_dict:
            for ann in json_dict["annotations"]:
                action_ids = ann["action_ids"]
                one_hot = np.zeros(81, dtype=np.bool)
                one_hot[action_ids] = True
                packed_act = np.packbits(one_hot[1:])
                clip2ann[ann["image_id"]].append(dict(bbox=ann["bbox"], packed_act=packed_act))

        movies_size = {}
        clips_info = {}
        for img in json_dict["images"]:
            mov = img["movie"]
            if mov not in movies_size:
                movies_size[mov] = [img["width"], img["height"]]
            clips_info[img["id"]] = [mov, img["timestamp"]]
        self.movie_info = NpInfoDict(movies_size, value_type=np.int32)
        clip_ids = sorted(list(clips_info.keys()))

        if remove_clips_without_annotations:
            clip_ids = [clip_id for clip_id in clip_ids if clip_id in clip2ann]


        # clip2ann: {image_id: [{bbox, packed_act},...], ...}
        # clip_ids: [image_id, ...]
        self.anns = NpBoxDict(clip2ann, clip_ids, value_types=[("bbox", np.float32), ("packed_act", np.uint8)])

        clips_info = {
            clip_id:
                [
                    self.movie_info.convert_key(clips_info[clip_id][0]), # movie_index
                    clips_info[clip_id][1] # timestamp
                ] for clip_id in clip_ids
        }
        self.clips_info = NpInfoDict(clips_info, value_type=np.int32)


    def __getitem__(self, idx):

        _, clip_info = self.clips_info[idx]

        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info[mov_id]
        video_data = self._decode_video_data(movie_id, timestamp)

        im_w, im_h = movie_size

        boxes, packed_act = self.anns[idx]

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # guard against no boxes
        # num_boxes, 4
        boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xywh").convert("xyxy")

        # Decode the packed bits from uint8 to one hot, since AVA has 80 classes,
        # it can be exactly denoted with 10 bytes, otherwise we may need to discard some bits.
        one_hot_label = np.unpackbits(packed_act, axis=1)
        # num_boxes, 80
        one_hot_label = torch.as_tensor(one_hot_label, dtype=torch.uint8)

        boxes.add_field("labels", one_hot_label)


        boxes = boxes.clip_to_image(remove_empty=True)
        # extra fields
        extras = {}

        if self.transforms is not None:
            video_data, boxes, transform_randoms = self.transforms(video_data, boxes)
            slow_video, fast_video = video_data
            h, w = slow_video.shape[-2:]
            whwh = torch.tensor([w,h,w,h], dtype=torch.float32)

            # add infos neccessary for memory feature
            extras["movie_id"] = movie_id
            extras["timestamp"] = timestamp

            return slow_video, fast_video, whwh, boxes, extras, idx

        return video_data, boxes, idx, movie_id, timestamp

    def return_null_box(self, im_w, im_h):
        return BoxList(torch.zeros((0, 4)), (im_w, im_h), mode="xyxy")


    def get_video_info(self, index):
        _, clip_info = self.clips_info[index]
        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info[mov_id]
        w, h = movie_size
        return dict(width=w, height=h, movie=movie_id, timestamp=timestamp)


    def _decode_video_data(self, dirname, timestamp):
        # decode target video data from segment per second.

        video_folder = os.path.join(self.video_root, dirname)
        right_span = self.frame_span//2
        left_span = self.frame_span - right_span

        #load right
        cur_t = timestamp
        right_frames = []
        while len(right_frames)<right_span:
            video_path = os.path.join(video_folder, "{}.mp4".format(cur_t))
            # frames = cv2_decode_video(video_path)
            frames = av_decode_video(video_path)
            if len(frames)==0:
                raise RuntimeError("Video {} cannot be decoded.".format(video_path))
            right_frames = right_frames+frames
            cur_t += 1

        #load left
        cur_t = timestamp-1
        left_frames = []
        while len(left_frames)<left_span:
            video_path = os.path.join(video_folder, "{}.mp4".format(cur_t))
            # frames = cv2_decode_video(video_path)
            frames = av_decode_video(video_path)
            if len(frames)==0:
                raise RuntimeError("Video {} cannot be decoded.".format(video_path))
            left_frames = frames+left_frames
            cur_t -= 1

        #adjust key frame to center, usually no need
        min_frame_num = min(len(left_frames), len(right_frames))
        frames = left_frames[-min_frame_num:] + right_frames[:min_frame_num]

        video_data = np.stack(frames)

        return video_data

    def __len__(self):
        return len(self.clips_info)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Video Root Location: {}\n'.format(self.video_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str