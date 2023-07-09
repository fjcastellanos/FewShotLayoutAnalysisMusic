

AUGMENTATION_ALL = "all"
AUGMENTATION_NONE = "none"
AUGMENTATION_FLIPH = "flipH"
AUGMENTATION_FLIPV = "flipV"
AUGMENTATION_WHITE_BALANCE = "wb"
AUGMENTATION_ROTATION = "rot"
AUGMENTATION_SCALE = "scale"
AUGMENTATION_BLURING = "blur"
AUGMENTATION_DROPOUT = "drop"
AUGMENTATION_OVEREXPOSITION = "expos"
AUGMENTATION_RANDOM = "random"


AUGMENTATION_CHOICES = [
                AUGMENTATION_ALL,
                AUGMENTATION_NONE,
                AUGMENTATION_FLIPH,
                AUGMENTATION_FLIPV,
                AUGMENTATION_ROTATION,
                AUGMENTATION_SCALE, 
                AUGMENTATION_DROPOUT
                ]


AUGMENTATION_CHOICES_TRAIN = [
                AUGMENTATION_ALL,
                AUGMENTATION_NONE,
                AUGMENTATION_FLIPH,
                AUGMENTATION_FLIPV,
                AUGMENTATION_ROTATION,
                AUGMENTATION_SCALE,
                AUGMENTATION_RANDOM
                ]

AUGMENTATION_CHOICES_TEST = [
                AUGMENTATION_ALL,
                AUGMENTATION_NONE,
                AUGMENTATION_FLIPH,
                AUGMENTATION_FLIPV,
                AUGMENTATION_ROTATION,
                AUGMENTATION_SCALE, 
                AUGMENTATION_DROPOUT
                ]




kPIXEL_VALUE_FOR_MASKING = -1
kNUMBER_CHANNELS = 3


KEY_RESULT="result"
