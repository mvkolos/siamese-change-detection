from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightness,  RandomContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, DualTransform, RandomGamma
)


train_augmentations = [
    Flip(p=0.4),
    OneOf([
            RandomRotate90(p=1),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=1e-2, rotate_limit=45, p=1)], p=0.4),
        RandomBrightness(limit=0.08, p=0.2),

        
    OneOf([
       IAAAdditiveGaussianNoise(scale = 1.1, p=1),
        GaussNoise(var_limit = (10,20), p=1)
    ], p=0.4),
    
    
    OneOf([
            MotionBlur(p=1),
            MedianBlur(blur_limit=3, p=1),
            Blur(blur_limit=3, p=1),
        ], p=0.4),
]

train_augmenter = Compose(train_augmentations, p=1)


# new targets function
def new_tf_targets(self):
    def augment_images(images, **params):
        aug_images = [self.apply(image, **params) for image in images]
        
        return aug_images
        
    def apply_to_mask_(mask, **params):
        aug = self.apply_to_mask(mask)
        
        if aug.ndim==2:
            aug = aug[:,:,None]
        return aug
    
    return {
        'image': self.apply, # do not rename this one
        'image2': self.apply, # do not rename this one
        'mask': apply_to_mask_,
        'bboxes': self.apply_to_bboxes
    }

# redefine targets function
# don't forget to wrap it in property
DualTransform.targets = property(new_tf_targets)