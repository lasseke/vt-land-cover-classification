'''
Some helper functions needed across sub-packages.
'''

import sys
import glob
import configparser
import logging
import logging.handlers
from pathlib import Path


def get_plt_style_config_path() -> str:
    '''
    Reads name of *.mplstyle file for matplotlib plotting options specified
    in 'plotting_style_config.txt' and returns its full path.
    Must be present in '../src/visualization/mplstyles/'.
    '''

    # Config file
    config = configparser.ConfigParser()
    config.read('../plotting_style_config.txt')

    style_name = str(config['plots']['style'])

    # Retrieve valid options
    available_file_paths = glob.glob(
        '../src/visualization/mplstyles/*.mplstyle'
        )
    file_names = [Path(p).name for p in available_file_paths]

    if not style_name.endswith('.mplstyle'):
        style_name += '.mplstyle'

    if style_name in file_names:
        return f'../src/visualization/mplstyles/{style_name}'

    else:
        raise ValueError(
            f'''The file '{style_name}', defined in the
            'output_config.txt' file, does not exist! Available
            files: {[x for x in file_names]}
            '''
        )


class StreamToLogger(object):
    """
    Custom class to log all stdout and stderr streams.
    modified from:
    https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
    """

    def __init__(
        self, stream, logger, log_level=logging.INFO,
        also_log_to_stream=False
    ):
        self.logger = logger
        self.stream = stream
        self.log_level = log_level
        self.linebuf = ''
        self.also_log_to_stream = also_log_to_stream

    @classmethod
    def setup_stdout(cls, also_log_to_stream=True):
        """
        Setup logger for stdout
        """
        stdout_logger = logging.getLogger('STDOUT')
        sl = StreamToLogger(sys.stdout, stdout_logger, logging.INFO,
                            also_log_to_stream)
        sys.stdout = sl

    @classmethod
    def setup_stderr(cls, also_log_to_stream=True):
        """
        Setup logger for stdout
        """
        stderr_logger = logging.getLogger('STDERR')
        sl = StreamToLogger(sys.stderr, stderr_logger, logging.ERROR,
                            also_log_to_stream)
        sys.stderr = sl

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def setup_logging(log_file, log_level):
    """Setup logging to log to console and log file."""

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # setup log file
    one_mb = 1000000
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=one_mb,
        backupCount=10
    )

    fmt = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%y-%m-%d %H:%M:%S')

    handler.setFormatter(fmt)
    root_logger.addHandler(handler)

    # setup logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root_logger.addHandler(stream_handler)

    # redirect stdout/err to log file
    StreamToLogger.setup_stdout()
    StreamToLogger.setup_stderr()

    return root_logger


def kill_logging():
    logging.shutdown()
