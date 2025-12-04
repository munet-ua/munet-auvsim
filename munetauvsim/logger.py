"""
Logging configuration for AUV simulations.

Provides centralized logging setup with custom formatting, multiple handlers,
simulation time tracking, and flexible logger management. Supports console and
file output with independent configuration for main and communication logging.


Functions
---------
**Setup Functions:**

    setupMain(fileName, fileFormat, fileLevel, outFormat, outLevel)
        Configure and return main program logger.
    setupComm(name, fileName, file, out)
        Configure and return communication module logger.
    
**Logger Management:**

    addLog(name)
        Create logger that uses main logger handlers.
    noneLog(name)
        Create logger with no handlers (warnings only).
    removeLog(name)
        Remove logger and close unshared handlers.
    deepRemoveLog(name)
        Remove logger and close all its handlers.
    
**Handler Management:**

    addMainHandlers(subLog)
        Add main logger handlers to sublevel logger.
    removeHandlers(name)
        Remove all handlers from logger, closing unshared ones.
    closeHandler(handler)
        Close handler and update global variables.
    deepRemoveHandler(handler)
        Remove handler from all loggers and close it.

**Custom Features:**

    customRecordFactory(args, kwargs)
        Add simulation time field to log records.
    CustomFormatter
        Format log records with bracketed function names and multi-line support.

        
Global Variables
----------------
log : logging.Logger
    Main simulation logger instance.
consoleHandler : logging.StreamHandler
    Shared console output handler.
fileHandler : logging.FileHandler
    Shared file output handler.
simTime : str
    Current simulation time for log records (format: "MM:SS").

    
Notes
-----
The module uses a custom log record factory to inject simulation time into all
log records. Update simTime via direct assignment before logging. This is done
by the Simulator in each iteration loop by the _simulateX methods.
"""

from typing import Optional
from datetime import datetime
import logging
import os

#-----------------------------------------------------------------------------#

# Logging levels
DEBUG = logging.DEBUG           # 10
INFO = logging.INFO             # 20
WARNING = logging.WARNING       # 30
ERROR = logging.ERROR           # 40
CRITICAL = logging.CRITICAL     # 50

# Log record component formats
SIMTIME = '%(simTime)8s'
DATETIME = '%(asctime)s'
NAME  = '%(name)-8s'
LEVEL = '%(levelname)-7s'
MODULE = '%(module)-14s'
FUNCTION = '%(funcName)s'
MESSAGE = '%(message)s'

# Delimiter strings
C = ':'         # Colon
CS = ' : '      # Colon with spaces
D = '.'         # Dot
DA = '-'        # Dash
DAS = ' - '     # Dash with spaces
LAB = '<'       # Left angle bracket
RAB = '>'       # Right angle bracket
LSB = '['       # Left square bracket
RSB = ']'       # Right square bracket
P = '|'         # Pipe
S = ' '         # Space
E = ''          # Empty

# Formatting strings
FMT_DATE = '%M:%S'
FMT_OUT = P+SIMTIME+P+S+NAME+CS+LEVEL+S+RAB+S+MESSAGE
FMT_FILE = P+SIMTIME+S+DATETIME+P+S+NAME+S+LEVEL+S+FUNCTION+CS+MESSAGE

# Main logger name
MAIN_LOG = 'mnAUVsim'

# Global variables -----------------------------------------------------------#

# Main logger and main handlers
log = None
consoleHandler = None
fileHandler = None

# Register loggers needing main handlers
pending = []

# Custom logging
oldFactory = logging.getLogRecordFactory()  # Cache for original record factory
simTime = '0.00'                            # Initial value of custom field

###############################################################################

class CustomFormatter(logging.Formatter):
    """
    Custom log formatter with bracketed function names and multi-line support.
    
    Wraps function names in brackets with padding and preserves log prefix
    formatting when messages contain newline characters.
    

    Parameters
    ----------
    fmt : str, optional
        Log record format string.
    datefmt : str, optional
        Date/time format string.
    """

    def __init__(self, fmt:str=None, datefmt:str=None)->None:
        """Initialize formatter with parent class."""
        super().__init__(fmt, datefmt)

    def format(self, record):
        """
        Apply custom formatting to log record.
        

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format.

             
        Returns
        -------
        formatted : str
            Formatted log message string.
            

        Notes
        -----
        Transformations:

        - Function name wrapped in brackets: [funcName] with 19-char width
        - Multi-line messages: Log prefix inserted at each newline
        """

        # Wrap function name in brackets with white space padding after bracket
        if not (record.funcName.startswith("[")):
            func = f"[{record.funcName}]"
            record.funcName = f"{func:19}"

        # Insert log prefix when any newline characters are used in log message
        newline = '\n'
        if (newline in record.msg):
            # Make local copy of log record
            record = logging.makeLogRecord(record.__dict__)
            # Get log prefix format and ignore the rest
            prefixFmt, _, _ = self._fmt.partition(MESSAGE)
            # Add missing log record attributes if part of prefix
            if (DATETIME in prefixFmt):
                if (self.datefmt):
                    record.asctime = self.formatTime(record, self.datefmt)
                else:
                    record.asctime = self.formatTime(record)
            # Build actual log prefix from record
            prefix = prefixFmt % record.__dict__
            # Rebuild log message to include prefix on newlines
            lines = record.msg.split(newline)
            msg = (newline + prefix).join(lines)
            record.msg = msg

        return super().format(record)

###############################################################################

def customRecordFactory(*args, **kwargs):
    """
    Create log record with custom simTime field.
    

    Returns
    -------
    record : logging.LogRecord
        Log record with simTime attribute from global simTime variable.

         
    Notes
    -----
    - Wraps original log record factory to inject simulation time.
    - Set via: logging.setLogRecordFactory(customRecordFactory)
    """

    record = oldFactory(*args, **kwargs)
    record.simTime = simTime
    return record

###############################################################################

def addMainHandlers(subLog:logging.Logger)->None:
    """
    Add main logger handlers (console, file) to sublevel logger.
    

    Parameters
    ----------
    subLog : logging.Logger
        Logger to receive main handlers.

         
    Notes
    -----
    - Only adds handlers that exist (not None).
    - Logs debug message on successful activation.
    """
    
    # Add main logger standard out handler
    if (consoleHandler is not None):
        subLog.addHandler(consoleHandler)
    # Add main logger file handler
    if (fileHandler is not None):
        subLog.addHandler(fileHandler)
    subLog.debug('%s logger activated', subLog.name)

###############################################################################

def setupMain(fileName:Optional[str] = MAIN_LOG+'.log',
              fileFormat:Optional[str] = FMT_FILE,
              fileLevel:int = DEBUG,
              outFormat:Optional[str] = FMT_OUT,
              outLevel:int = INFO,
              )->logging.Logger:
    """
    Configure and return main program logger with console and file handlers.
    

    Parameters
    ----------
    fileName : str, default='mnAUVsim.log'
        Log file name. If None, file output disabled.
    fileFormat : str, optional
        Format string for file handler. If None, file output disabled.
    fileLevel : int, default=DEBUG
        Minimum log level for file handler.
    outFormat : str, optional
        Format string for console handler. If None, console output disabled.
    outLevel : int, default=INFO
        Minimum log level for console handler.

         
    Returns
    -------
    log : logging.Logger
        Main logger instance with configured handlers.
        

    Notes
    -----
    - Sets custom log record factory for simulation time field.
    - Processes pending loggers that were created before main logger setup.
    - Global variables updated: log, consoleHandler, fileHandler.
    """

    global log, consoleHandler, fileHandler, pending

    if (log is None):

        # Create main logger
        logging.setLogRecordFactory(customRecordFactory)
        log = logging.getLogger(MAIN_LOG)
        log.setLevel(DEBUG)

        # Create standard out console handler
        if (outFormat is not None):
            if (consoleHandler is None):
                consoleHandler = logging.StreamHandler()
                consoleHandler.set_name('Console handler')
                consoleHandler.setLevel(outLevel)
                consoleHandler.setFormatter(CustomFormatter(outFormat))
            log.addHandler(consoleHandler)
            log.info('Console logging started')

        # Create main file handler
        if (fileFormat is not None):
            if (fileHandler is None):
                fileHandler = logging.FileHandler(fileName)
                fileHandler.set_name('File handler')
                fileHandler.setLevel(fileLevel)
                fileHandler.setFormatter(CustomFormatter(fileFormat,
                                                           FMT_DATE))
            log.addHandler(fileHandler)
            start = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            log.info('File logging started at %s in %s', 
                     start, os.path.basename(fileName))
        
        # Add main handlers to roster of pending loggers
        while pending:
            name = pending.pop()
            modLog = logging.getLogger(name)
            addMainHandlers(modLog)

    return log

###############################################################################

def addLog(name:str)->logging.Logger:
    """
    Create logger that shares main logger handlers.
    

    Parameters
    ----------
    name : str
        Logger name.

          
    Returns
    -------
    logger : logging.Logger
        New or existing logger with main handlers.

          
    Notes
    -----
    - If main logger not yet created, logger is added to pending list.
    - Returns existing logger if name already registered.
    """

    global pending

    # Check if logger already exists
    if (name not in logging.Logger.manager.loggerDict):

        # Create new logger
        thisLog = logging.getLogger(name)
        thisLog.setLevel(DEBUG)

        # Register logger if main logger not yet created
        if (log is None):
            pending.append(name)

        # Otherwise, add main logger handlers
        else:
            addMainHandlers(thisLog)

        return thisLog
    
    # Return existing logger
    else:
        return logging.getLogger(name)

###############################################################################

def noneLog(name:str)->logging.Logger:
    """
    Create or configure logger with no handlers.
    

    Parameters
    ----------
    name : str
        Logger name.

          
    Returns
    -------
    logger : logging.Logger
        Logger with no handlers. WARNING+ messages go to stderr.

         
    Notes
    -----
    - If logger exists: Removes/closes non-shared handlers.
    - If logger is main logger: Closes all handlers.
    - Sets level to WARNING.
    """

    global log

    # Get logger
    thisLog = logging.getLogger(name)
    thisLog.setLevel(WARNING)

    # Check if logger has handlers
    if (thisLog.hasHandlers()):
        # Logger is the main logger
        if (thisLog is log):
            while thisLog.handlers:
                deepRemoveHandler(thisLog.handlers[0])
        # Logger is not the main logger
        else:
            removeHandlers(name)

    # Set main logger
    if (name is MAIN_LOG):
        log = thisLog

    return thisLog

###############################################################################

def closeHandler(handler:logging.Handler)->None:
    """
    Close handler and update global handler variables.
    

    Parameters
    ----------
    handler : logging.Handler
        Handler to close.

           
    Notes
    -----
    If handler is consoleHandler or fileHandler, sets global to None.
    """

    global consoleHandler
    global fileHandler

    # Close handler
    handler.close()

    # Set value to None if handler is a main logger handler
    if (handler is consoleHandler):
        consoleHandler = None
    elif (handler is fileHandler):
        fileHandler = None

###############################################################################

def removeHandlers(name:str)->None:
    """
    Remove all handlers from logger, closing unshared ones.
    

    Parameters
    ----------
    name : str
        Logger name.
        

    Notes
    -----
    - Iterates through all handlers and removes them.
    - Closes handlers not shared with other loggers.
    """
    
    # Get logger
    thisLog = logging.getLogger(name)

    # Remove handlers from logger
    while thisLog.handlers:
        handler = thisLog.handlers[0]
        log.debug('Removing %s from %s',handler.get_name(),name)
        thisLog.removeHandler(handler)

        # Close handler if not shared with other loggers
        closeIt = True
        for _,l in logging.Logger.manager.loggerDict.items():
            if isinstance(l, logging.Logger):
                if (handler in l.handlers):
                    closeIt = False
                    break
        if (closeIt):
            log.debug('Closing %s',handler.get_name())
            closeHandler(handler)

###############################################################################

def deepRemoveHandler(handler:logging.Handler)->None:
    """
    Remove handler from all loggers and close it.
    

    Parameters
    ----------
    handler : logging.Handler
        Handler to remove and close.

         
    Notes
    -----
    - Searches all registered loggers for this handler.
    - Removes from each logger found, then closes handler.
    """

    # Step over all known loggers
    for thisLogName,thisLog in logging.Logger.manager.loggerDict.items():
        # Check if is a Logger
        if isinstance(thisLog, logging.Logger):
            # Remove the handler if it exists in the logger
            if (handler in thisLog.handlers):
                log.debug('Removing %s from %s',handler.get_name(),thisLogName)
                thisLog.removeHandler(handler)
    
    # Close handler
    log.debug('Closing %s',handler.get_name())
    closeHandler(handler)

###############################################################################

def removeLog(name:str)->None:
    """
    Remove logger and close unshared handlers.
    

    Parameters
    ----------
    name : str
        Logger name to remove.
        

    Notes
    -----
    - Handlers shared with other loggers are not closed.
    - If removing main logger, sets global log to None.
    """

    global log

    # Get logger
    thisLog = logging.getLogger(name)
    log.debug('Removing logger %s and closing any sole handlers',name)

    # Remove handlers from logger
    removeHandlers(name)
    
    # Remove the logger
    del logging.Logger.manager.loggerDict[name]
    if (thisLog is log):
        log = None

###############################################################################

def deepRemoveLog(name:str)->None:
    """
    Remove logger and close all its handlers.
    

    Parameters
    ----------
    name : str
        Logger name to remove.

         
    Notes
    -----
    - Closes all handlers regardless of sharing with other loggers.
    - If removing main logger, sets global log to None.
    """

    global log

    # Get logger
    thisLog = logging.getLogger(name)
    log.debug('Removing logger %s and closing all associated handlers')

    # Remove logger handlers from all loggers
    while thisLog.handlers:
        deepRemoveHandler(thisLog.handlers[0])

    # Remove the logger
    del logging.Logger.manager.loggerDict[name]
    if (thisLog is log):
        log = None

###############################################################################

def setupComm(name:str = 'comm',
              fileName:Optional[str] = 'comm.log',
              file:bool = True,
              out:bool = True,
              )->logging.Logger:
    """
    Configure and return communication module logger.
    

    Parameters
    ----------
    name : str, default='comm'
        Logger name.
    fileName : str, default='comm.log'
        Communication log file name.
    file : bool, default=True
        Enable separate communication log file.
    out : bool, default=True
        Enable console output for communication logs.

         
    Returns
    -------
    commLog : logging.Logger
        Communication logger with configured handlers.

        
    Notes
    -----
    - If the main logger has the console handler turned off, then the comms
      logger will not print to the console even if turned on.
    - If the comms logger is set to print to its own log file, then it will not
      print log records in the main logger file.
    """

    # Create / fetch logger
    commLog = logging.getLogger(name)
    commLog.setLevel(DEBUG)

    # Add main logger standard out console handler
    if ((out) and (consoleHandler is not None)):
        commLog.addHandler(consoleHandler)

    # Add independent file handler
    if (file):
        commFileHandler = logging.FileHandler(fileName)
        commFileHandler.set_name('Comms file handler')
        # Inherit logging level from main file handler
        if (fileHandler is not None):
            commFileHandler.setLevel(fileHandler.level)
        else:
            commFileHandler.setLevel(DEBUG)
        commFileHandler.setFormatter(CustomFormatter(FMT_FILE, FMT_DATE))
        commLog.addHandler(commFileHandler)
        start = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        commLog.info('Comms file logging started at %s in %s',
                     start, os.path.basename(fileName))
    # Otherwise add main logger file handler
    else:
        if (fileHandler is not None):
            commLog.addHandler(fileHandler)

    commLog.info('%s logger activated', name)
    return commLog

###############################################################################