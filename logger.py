
import logging
# import auxiliary_module

def loggerGenerator(name=None, level=logging.DEBUG, fileName='log.log'):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create logger with 'spam_application'
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)

        # create file handler which logs even debug messages
    fh = logging.FileHandler(fileName)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(formatter)