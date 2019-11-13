
import logging
# import auxiliary_module

def loggerGenerator(name=None, level=logging.DEBUG, fileName='log.log'):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create logger with 'spam_application'
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)