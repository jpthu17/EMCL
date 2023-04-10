import torch
from torch.utils.data import DataLoader
from .dataloader_msrvtt_retrieval import MSRVTTDataset
from .dataloader_activitynet_retrieval import ActivityNetDataset
from .dataloader_didemo_retrieval import DiDeMoDataset
from .dataloader_lsmdc_retrieval import LsmdcDataset
from .dataloader_msvd_retrieval import MsvdDataset


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTTDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTTDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_testset)
    except:
        test_sampler = None  # cpu
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LsmdcDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler


def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LsmdcDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_testset)
    except:
        test_sampler = None  # cpu
    dataloader_lsmdc = DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_lsmdc, len(lsmdc_testset)


def dataloader_activity_train(args, tokenizer):
    activity_dataset = ActivityNetDataset(
        subset="train",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler


def dataloader_activity_test(args, tokenizer, subset="test"):
    activity_testset = ActivityNetDataset(
        subset=subset,
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(activity_testset)
    except:
        test_sampler = None  # cpu
    dataloader_activity = DataLoader(
        activity_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_activity, len(activity_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MsvdDataset(
        subset="train",
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MsvdDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    dataloader_msvd = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msvd, len(msvd_testset)


def dataloader_didemo_train(args, tokenizer):
    didemo_dataset = DiDeMoDataset(
        subset="train",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler


def dataloader_didemo_test(args, tokenizer, subset="test"):
    didemo_testset = DiDeMoDataset(
        subset=subset,
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(didemo_testset)
    except:
        test_sampler = None  # cpu
    dataloader_didemo = DataLoader(
        didemo_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_didemo, len(didemo_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train": dataloader_msrvtt_train, "val": dataloader_msrvtt_test, "test": None}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd_train, "val":dataloader_msvd_test, "test":dataloader_msvd_test}
DATALOADER_DICT["lsmdc"] = {"train":dataloader_lsmdc_train, "val":dataloader_lsmdc_test, "test":dataloader_lsmdc_test}
DATALOADER_DICT["activity"] = {"train":dataloader_activity_train, "val":dataloader_activity_test, "test":None}
DATALOADER_DICT["didemo"] = {"train":dataloader_didemo_train, "val":None, "test":dataloader_didemo_test}