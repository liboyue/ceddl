import torch
import torchvision


def _load_from_dataset(dataset, rank, world_size, batch_size, pin_memory=True, shuffle=True, num_workers=0):

    sampler = torch.utils.data.distributed \
                .DistributedSampler(dataset,
                                    rank=rank,
                                    num_replicas=world_size,
                                    shuffle=shuffle)

    loader = torch.utils.data \
                .DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            pin_memory=pin_memory,
                            num_workers=num_workers)

    return loader


def load_cifar10(rank, world_size, batch_size, shuffle=True, num_workers=0):

    from torchvision.datasets import CIFAR10
    import torchvision.transforms as tf

    transform = tf.Compose([tf.ToTensor(),
                            tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    download = rank == 0
    train_data = CIFAR10(root='~/data', train=True, transform=transform, download=download)
    val_data = CIFAR10(root='~/data', train=False, transform=transform, download=False)

    train_loader = _load_from_dataset(train_data, rank, world_size, batch_size, 
                                      shuffle=shuffle, num_workers=num_workers)
    val_loader = _load_from_dataset(val_data, rank, world_size, batch_size, shuffle=False,
                                    num_workers=num_workers) # Don't shuffle val set

    return train_loader, val_loader


def load_mnist(rank, world_size, batch_size, shuffle=True, num_workers=0):

    from torchvision.datasets import MNIST
    import torchvision.transforms as tf

    transform = tf.Compose([tf.ToTensor(),
                            tf.Normalize((0.1307,), (0.3081,))])
    download = rank == 0
    train_data = MNIST('~/data', train=True, transform=transform, download=download)
    val_data = MNIST('~/data', train=False, transform=transform)

    train_loader = _load_from_dataset(train_data, rank, world_size, batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = _load_from_dataset(val_data, rank, world_size, batch_size, shuffle=False,
                                    num_workers=num_workers)

    return train_loader, val_loader


def load_imagenet(rank, world_size, batch_size, data_path='~/data/imagenet', num_workers=0):

    import os
    from opencv_transforms import transforms as tf
    import cv2

    class ImageFolder(torchvision.datasets.ImageFolder):

      def __getitem__(self, index):

        image_path, target = self.samples[index]
        im = cv2.imread(image_path)

        if self.transform is not None:
            sample = self.transform(im)

        return sample, target


    train_transform = tf.Compose([
        tf.RandomResizedCrop(224),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # For RGB
        tf.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]) # For BGR
    ])

    val_transform = tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
    val_data = ImageFolder(os.path.join(data_path, 'val'), transform=val_transform)
    
    train_loader = _load_from_dataset(train_data, rank, world_size, batch_size, shuffle=True,
                                      num_workers=num_workers)
    val_loader = _load_from_dataset(val_data, rank, world_size, batch_size, shuffle=False,
                                    num_workers=num_workers)

    return train_loader, val_loader
