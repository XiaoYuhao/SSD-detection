import logging
import logging.handlers
import time

def getLogger(log_name):
    logger = logging.getLogger("logger")
    #time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    filepath = "log/%s.log" % log_name
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=filepath, mode='a')

    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.DEBUG)
    handler2.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger