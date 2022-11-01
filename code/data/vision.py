import os
import torch
import torch.utils.data as data


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, transform=None):
        # has_transforms = transforms is not None
        # has_separate_transform = transform is not None
        # if has_transforms and has_separate_transform:
        #     raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        # self.target_transform = target_transform

        # if has_separate_transform:
        transforms = StandardTransform(transform, None) #target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
        # head = "Dataset " + self.__class__.__name__
        # body = ["Number of datapoints: {}".format(self.__len__())]
        # body += self.extra_repr().splitlines()
        # if hasattr(self, "transforms") and self.transforms is not None:
        #     body += [repr(self.transforms)]
        # lines = [head] + [" " * self._repr_indent + line for line in body]
        # return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        raise NotImplementedError
        # lines = transform.__repr__().splitlines()
        # return (["{}{}".format(head, lines[0])] +
        #         ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        raise NotImplementedError
        # return ""


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        raise NotImplementedError
        # lines = transform.__repr__().splitlines()
        # return (["{}{}".format(head, lines[0])] +
        #         ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        raise NotImplementedError
        # body = [self.__class__.__name__]
        # if self.transform is not None:
        #     body += self._format_transform_repr(self.transform,
        #                                         "Transform: ")
        # if self.target_transform is not None:
        #     body += self._format_transform_repr(self.target_transform,
        #                                         "Target transform: ")

        # return '\n'.join(body)