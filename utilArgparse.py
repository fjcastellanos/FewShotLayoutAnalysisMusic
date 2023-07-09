import argparse
import utilConst

class DefaultListActionAugmentation(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values:
            for value in values:
                if value not in utilConst.AUGMENTATION_CHOICES:
                    message = ("invalid choice: {0!r} (choose from {1})"
                               .format(value,
                                       ', '.join([repr(action)
                                                  for action in utilConst.AUGMENTATION_CHOICES])))

                    raise argparse.ArgumentError(self, message)
            setattr(namespace, self.dest, values)