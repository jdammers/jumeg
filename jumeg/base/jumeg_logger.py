#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 18.03.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------
import sys,os,time,re
import inspect
from distutils.dir_util import mkpath
import logging

'''
Example A:
--------

#--- in your main-script
#--- first import and setup a logger at the top
import logging
logger = logging.getLogger('root')

#---
from jumeg.base import jumeg_logger

# ...
# put your code e.g.: class XYZ() def XYZ()
# use logger e.g: logger.info() logger.debug() ...
# ...

__main__

opt = None

#--- or using argparser
#--- examples in jumeg.tools
# opt, parser = get_args(argv)


#--- set <logger> to the logger-obj generated in script
jumeg_logger.setup_script_logging(name=argv[0],opt=opt,logger=logger)
#  do your stuff


Example B classic from logging:
--------
import logging
logger = logging.getLogger('root')
logger.setLevel('DEBUG')
logger.addHandler(jumeg_loglog.LogFileHandler())

#--- change loglevel and format for stream
HNDLStream = jumeg_loglog.LogStreamHandler()
HNDLStream.setLevel(logging.WARNING)
logger.addHandler(HNDLStream)

logger.debug("Start LOG")
logger.info("Start LOG")
logger.warning("Start LOG")
logger.error("Start LOG")
'''


'''
https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

and this would look something like:

log = logging.getLogger('foobar')
sys.stdout = LoggerWriter(log.debug)
sys.stderr = LoggerWriter(log.warning)



class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)

    def flush(self):
        pass
        
class LoggerWriter(object):
    def __init__(self, writer):
        self._writer = writer
        self._msg = ''

    def write(self, message):
        self._msg = self._msg + message
        while '\n' in self._msg:
            pos = self._msg.find('\n')
            self._writer(self._msg[:pos])
            self._msg = self._msg[pos+1:]

    def flush(self):
        if self._msg != '':
            self._writer(self._msg)
            self._msg = ''
            
            
import logging
from contextlib import redirect_stdout

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.write = lambda msg: logging.info(msg) if msg != '\n' else None

with redirect_stdout(logging):
    print('Test')
#------
import logging
from contextlib import redirect_stdout


logger = logging.getLogger('Meow')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='[{name}] {asctime} {levelname}: {message}',
    datefmt='%m/%d/%Y %H:%M:%S',
    style='{'
)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.write = lambda msg: logger.info(msg) if msg != '\n' else None

with redirect_stdout(logger):
    print('Test')
    
    
    
import logging
import sys

class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
       self.logger = logger
       self.log_level = log_level
       self.linebuf = ''

   def write(self, buf):
       for line in buf.rstrip().splitlines():
           self.logger.log(self.log_level, line.rstrip())

logging.basicConfig(
   level=logging.DEBUG,
   format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
   filename="out.log",
   filemode='a'
)

stdout_logger = logging.getLogger('STDOUT')
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

stderr_logger = logging.getLogger('STDERR')
sl = StreamToLogger(stderr_logger, logging.ERROR)
sys.stderr = sl

print "Test to standard out"
raise Exception('Test to standard error')
'''

__version__="2019.05.10.001"



#===========================================================
#=== test logging stdout, stderr
#===========================================================
def print_to_logger(raw_fname,raw=None,**cfg):
  
   #--- log stdout,stderr
   jumeg_logger.log_stdout(label=" LOGTEST")
   jumeg_logger.log_stderr()
   
   print("  -> TEST1 print to logger: {}".format(raw_fname) )

  #--- return back stdout/stderr from logger
   jumeg_logger.log_stdout(reset=True)
   jumeg_logger.log_stderr(reset=True)
   
   return raw_fname,raw


class StreamLoggerSTD(object):
   """
   coppy from:
   https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
   Fake file-like stream object that redirects writes to a logger instance.
   
   :param logger:  logger obj <None>
   :param logname:  <"root">
   :param level:    <logging.info>
   :param label:    <None>
   
   Example
   -------
    sys.stdout  = StreamLoggerSTD(logname="root",level=logging.INFO,label="TESTLOG OUT")
    sys.stderr = StreamLoggerSTD(logname="root",level=logging.INFO,label="TESTLOG ERR")
   
   """
   def __init__(self,logger=None,logname="root",level=logging.INFO,label=None):
       """
       
       :param logger:
       :param logname:
       :param level:
       :param label:
       """
   
       if not logger:
          self.logger=logging.getLogger(logname)
       else:
          self.logger = logger
       
       self.loglevel = level
       self._label   = label
       
       #stack = inspect.stack()
       #self._log_label = stack[0][3]
       #self.depth = len( stack )+ function_depth
       #self.loglevel= level
       #self._save_std = sys.stdout
       #sys.stdout = self
   
   @property
   def label(self): return self._label
   @label.setter
   def label(self,v):
       self._label=v
       
   @staticmethod
   def __get_call_info():
       """
       https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
       """
       stack = inspect.stack()
    
       # stack[1] gives previous function ('info' in our case)
       # stack[2] gives before previous function and so on
    
       fn = stack[2][1]
       ln = stack[2][2]
       func = stack[2][3]
    
       return fn,func,ln
   
   def __get_infolevel(self):
       msg   = []
       stack = inspect.stack()
       idx   = len(stack) - 1
       lnr   = 0
       while idx > len(stack): # - self.depth:
           if idx > 0:
               msg.append(stack[idx][3])
               lnr = stack[idx][2]
           idx -= 1
       msg.append(str(lnr))
       return( ".".append(msg) )

   def write(self, buf):
       if re.match(r'^\s*$',buf):
          return
          
       if self.label:
          self.logger.log(self.loglevel,self.label + " : "+ buf.rstrip())
       else:
          self.logger.log(self.loglevel, buf.rstrip())
       
       #msg="---->>>> " +f
       #msg += "\n-> " + str(sys._getframe(0)) + "\n"
       #msg += "-> " + str(sys._getframe(0).f_code.co_filename)
       #msg += "\n--> " + str(sys._getframe(1)) + "\n"
       #msg += "--> " + str(sys._getframe(1).f_code.co_filename)
       #msg += "\n---> " + str(sys._getframe(2)) + "\n"
       #msg += "---> " + str(sys._getframe(2).f_code.co_filename)

   def flush(self):
       pass

def log_stdout(reset=False,**kwargs):
    if reset:
       sys.stdout = sys.__stdout__
    else:
       sys.stdout = StreamLoggerSTD(**kwargs)
    return sys.stdout

def log_stderr(reset=False,**kwargs):
    if reset:
       sys.stderr = sys.__stderr__
    else:
       sys.stderr = StreamLoggerSTD(**kwargs)
    return sys.stderr

class JuMEGLogFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors
    https://stackoverflow.com/questions/14844970/modifying-logging-message-format-based-on-message-logging-level-in-python3
    
    Example:
    --------
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger_ch = logging.StreamHandler()
    logger_ch.setLevel(logging.INFO)
    logger_ch.setFormatter(JuMEGLogFormatter())
    logger.addHandler(logger_ch)
    """

    FORMATS = {
        logging.INFO:   "%(levelname)s - %(asctime)s — %(name)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s\n",
        #logging.INFO:   "\n%(levelname)s - %(asctime)s — %(module)s - %(funcName)s:%(lineno)d :\n%(message)s",
        logging.ERROR:  "\n%(levelname)s - %(asctime)s — %(name)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s",
        logging.WARNING:"\n%(levelname)s - %(asctime)s — %(name)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s",
        logging.DEBUG:  "%(levelname)s - %(asctime)s — %(name)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s\n"
    
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.DEBUG])
        fmt_date = "%Y-%m-%d %H:%M:%S"  #'%Y-%m-%dT%T%Z'
        formatter = logging.Formatter(log_fmt,fmt_date)
        return formatter.format(record)

class LogStreamHandler(logging.StreamHandler):
    def __init__(self,level=None):
        super().__init__()
        self._level = level if level else logging.NOTSET
        self.setFormatter(JuMEGLogFormatter())
    
    def setLevel(self, level):
        """
        set Handler logging level
        :param level:
        :return:
        """
        super().setLevel(level)
        self._level=level
        self.setFormatter(JuMEGLogFormatter())
        
    def getLevel(self):
        return self._level
    
class LogFileHandler(logging.FileHandler):
    
    def __init__(self,fname=None,prefix=None,name="root_logger",postfix="001",extention=".log",path=None,level=None,mode="a"):
        """
        LogFileHandler : for writing log messages into a file
      
        :param fname    : name of logfile
                          if fname is defined no <auto generated log filename> will be generated
        :param name     : name of script as part of <auto generated log filename>
        :param prefix   : prefix in log-filename  <None>
        :param postfix  : postfix in log-filename <001>
        :param extention: logfile extention       <.log>
        :param path     : logfile path            <None>
        :param level    : log level
        :param mode     : logfile mode <a> append [a,w]
        :param path     : logfile path            <None>
       
        :return
        LogFileHandler CLASS
        
        Example:
        --------
        import logging
        logger = logging.getLogger('root')
        logger.setLevel('DEBUG')

        HFile1=jumeg_logger.LogFileHandler(fname="logfile_test01.log")
        HFile1.setLevel(logging.DEBUG)
        logger.addHandler(HFile1)
        
        HFile2=jumeg_logger.LogFileHandler(prefix="test",name="meg",postfix="0815",level=logging.INFO)
        logger.addHandler(HFile2)
        
        logfilename if fname=None
        <prefix>_<name>_<user_name>_<date time>_<postfix><extention>
         M100_preprocessing_meguser_2019-03-18-10-01-01_001.log
         
        """
        self.filename = self.init_logfile_name(fname=fname,prefix=prefix,name=name,postfix=postfix,extention=extention,path=path)
       #--- mk log dir if not exists
        mkpath( os.path.dirname(self.filename) )
        
        super().__init__(self.filename,mode=mode)
        self._level = level if level else logging.NOTSET
        self.setFormatter(JuMEGLogFormatter())
    
    def setLevel(self, level):
        """
        set Handler logging level
        :param level:
        :return:
        """
        super().setLevel(level)
        self._level = level
        self.setFormatter(JuMEGLogFormatter())

    def getLevel(self):
        return self._level

    
    def init_logfile_name(self,fname=None,prefix=None,name="logger_info",postfix="001",extention=".log",path=None):
       """
       :param fname    : name of logfile
                         if fname is defined no <auto generated log filename> will be generated
       :param name     : name of script as part of <auto generated log filename> <logger_info>
       :param prefix   : prefix in log-filename  <None>
       :param postfix  : postfix in log-filename <001>
       :param extention: logfile extention       <.log>
       :param path     : logfile path            <None>
       
       :return:
        logger filename
        
       Example
       --------
       logfilename if fname=None
       <prefix>_<name>_><user>_<date-time>_<postfix><extention>
       fname_log = logger.init_logfile_name(prefix="M100",name="preprocessing",postfix="001",extention=".log")
       print(fname_log)
       M100_preprocessing_meguser_2019-03-18-10-01-01_001.log
       
       """
       if fname:
          if path:
             return os.path.join( os.path.abspath(path),fname)
          else:
             return fname
       
       fn = name+"_"+os.getenv("USER","meg")+"_"+ time.strftime("%G-%m-%d %H:%M:%S",time.localtime())+"_"+postfix+ extention
       if prefix:
          fn = prefix +"_"+ fn
       if path:
          fn = os.path.abspath(path) +"/"+ fn
       return fn.replace(" ","_").replace(":","-")

class LoggingContext(object):
    def __init__(self, logger, level=None, handler=None, close=True):
        """
        Using a context manager for selective logging, changing temporary the loglevel
        https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
        :param logger:
        :param level:
        :param handler:
        :param close:
        
        Example:
        --------
         logger = logging.getLogger('foo')
         logger.addHandler(logging.StreamHandler())
         logger.setLevel(logging.INFO)
         with LoggingContext(logger, level=logging.DEBUG):
              logger.debug('3. This should appear once on stderr.')
         logger.debug('4. This should not appear.')
         h = logging.StreamHandler(sys.stdout)
         with LoggingContext(logger, level=logging.DEBUG, handler=h, close=True):
             logger.debug('5. This should appear twice - once on stderr and once on stdout.')

        """
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions
        
def setup_script_logging(fname=None,name=None,opt=None,level="DEBUG",logger=None,path=None,version=None,logfile=False,mode="a",captureWarnings=True):
    """
    setup logger :
     loging level, log handler, logs information e.g input parameter
    
    :param fname  : name of logfile
                    if fname is defined no <auto generated log filename> will be generated
    :param name   : name of script as part of <auto generated log filename> if logfile==True
    :param opt    : option obj from argpraser
                    opt, parser = get_args(argv)
    
    :param level  : logging level <DEBUG>
    :param logger : logger <root logger obj> if <None> get new logger obj
    :param version: version  e.g.__version__
    :param logfile: flag log to logfile, will overwrite opt.logfile
    :param path   : logfile path <None> :
    :param mode   : logfile mode <a>  [a,w]  append or write
    :param captureWarnings: capture Warnings form wranings module <True>
    
    :return:
    logger
    
    """
    if not logger:
       logger=logging.getLogger("root")
   #----
    name = name if name else "jumeg_logfile"
    
    logger.setLevel(level)
    logging.captureWarnings(captureWarnings)
    
    HStream = LogStreamHandler()
    # HStream.setLevel(logging.INFO)
    HStream.setLevel( logger.getEffectiveLevel() )
    logger.addHandler(HStream)
    
    if fname:
       script_name = None
    else:
       script_name = os.path.basename(name)
       script_name = os.path.splitext(script_name)[0]
       
    if not version:
       try:
         #--- get __version__ from caller obj
          version = sys._getframe(1).f_globals.get("__version__") # get global version from caller
       except:
          version =__version__
          
    version = version if version else __version__
    
    msg=[]
    log2file = logfile
    
    if opt:
      # use logfile parameter; avoid logfile at start, write logfile for each file to process
      # try:
      #    log2file = opt.logfile
      # except:
      #   # logger.error("logfile option is not defined: no logfile will be generated")
      #    pass
       
       try:
          if opt.verbose:
             msg = ["---> {}".format(script_name),
                    "  -> version             : {}".format(version),
                    "  -> python sys version  : {}".format(sys.version_info),
                    "   " + "-" * 40," --> ARGV parameter:"]
             for k,v in sorted(vars(opt).items()):
                 msg.append("  -> {0:<30}: {1}".format(k,v))
             
       except:
          #logger.error("logfile option <verbose> is not defined: no parameter info")
          pass
        
    if log2file:
       HFile = LogFileHandler(fname=fname,name=script_name,mode=mode,path=path)
       HFile.setLevel(logging.DEBUG)
       logger.addHandler(HFile)
    
    if msg:
       logger.info("\n".join(msg) +"\n")

    return logger

    
def update_filehandler(logger=None,logger_name="root",**kwargs):
    """
    close all filehandlers first and add a new filehandler to the logger
    https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a
    
    :param logger      : the logger obj <None>
    :param logger_name : logger name <root>, used if logger is None
    
    parameter for LogFileHandler
    :param fname:
    :param prefix:
    :param name:
    :param postfix:
    :param extention:
    :param path:
    :param level:
    
    :return:
    new filehandler
    
    """
    
    if not logger:
       if logger_name:
           logger=logging.getLogger("root")
       logger = logging.getLogger()     # root logger - Good to get it only once.
    
    level=kwargs.get("level")
    
    for hdlr in logger.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr,logging.FileHandler):
           if not level:
              level = hdlr.getLevel() if hdlr.getLevel() else logger.getEffectiveLevel()
           hdlr.flush()
           hdlr.close()
           logger.removeHandler(hdlr)
           
    hdlr = LogFileHandler(**kwargs)
    logger.addHandler( hdlr ) # set the new handler
    return hdlr
   
#=========================================================================================
#==== MAIN
#=========================================================================================
def getLogger(name="root",captureWarnings=True,level="INFO"):
    """
    get a logger, calls logging.getLogger(), add a Stream- and a FileHandler to the logger
    
    :param name:             logger name <root>
    :param captureWarnings: capture and log printed warning messages <True>
    :param level:           log level <INFO>
     [NOTSET,INFO,DEBUG,WARNINGS/WARN,ERROR,CRITICAL] or logging.XYZ or [0,10,20,30,40,50]
    :return:
    logger obj
    
    Example:
    --------
    import logging
    from jumeg import jumeg_logger
    logger = jumeg_logger.getLogger(name='root')
    logger.setLevel(logging.INFO)
    
    HFile=jumeg_logger.LogFileHandler(fname="logfile_test01.log")
    HFile.setLevel(logging.DEBUG)
    logger.addHandler(HFile)

    HStream = jumeg_logger.LogStreamHandler()
    HStream.setLevel(logging.INFO)
    logger.addHandler(HStream)

    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warning message')
    logger.error('error message')
    
    # in sub module:
      import logging
      logger = logging.getLogger('root')
      
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(LogStreamHandler())
    logger.addHandler(LogFileHandler())
    logging.captureWarnings(captureWarnings)
    
    return logger



'''
#if __name__ == "__main__":
   #init_logger()


class _oldJuMEG_Logger(object):
    """
     logger cls
     :param: app_name => logger name <None>
     :param: level    => logging level  <10>
        level values:
         CRITICAL 50
         ERROR 	 40
         WARNING  30
         INFO 	 20
         DEBUG 	 10
         NOTSET 	 0

     https://realpython.com/python-logging/
     https://docs.python.org/3/howto/logging-cookbook.html
     https://stackoverflow.com/questions/44522676/including-the-current-method-name-when-printing-in-python

     Example:
     ---------
     from jumeg.jumeg_base import JuMEG_Logger
     myLog=JuMEG_Logger(app_name="MYLOG",level=logging.DEBUG)
     myLog.info( "test logging info instead of using <print> ")

     from jumeg.jumeg_base import JuMEG_Base_Basic as JB
     jb=JB()
     jb.Log.info("test logging info instead of using <print>")

     https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
     import logging
     logger = logging.getLogger('root')
     FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
     logging.basicConfig(format=FORMAT)
     logger.setLevel(logging.DEBUG)

    """
    
    def __init__(self,app_name=None,level=10,**kwargs):
        super(JuMEG_Logger,self).__init__(**kwargs)
        import logging
        self.verbose = False
        self.logger = logging.getLogger(app_name or __name__)
        self.logger.setLevel(logging.DEBUG)
        self.fmt_info = 'JuMEG LOG %(asctime)s %(message)s'
        logging.basicConfig(format=self.fmt_info,datefmt='%Y/%m/%d %I:%M:%S')
        
        #self.fmt_debug   = '%(asctime)-15s] %(levelname) %(funcName)s %(message)s'
        #self.fmt_error   = '[%(asctime)-15s] [%(levelname)08s] (%(funcName)s %(message)s'
        #formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    
    def list2str(self,msg):
        if isinstance(msg,(list)):
            return "\n" + "\n".join(msg)
        return msg
    
    def info(self,msg):
        #self.logger.setFormatter()
        self.logger.info(self.list2str(msg))
    
    def warning(self,msg):
        self.logger.warning(self.list2str(msg))
    
    def error(self,msg):
        if isinstance(msg,(list)):
            self.logger.error("\nERROR:\n" + self.list2str(msg) + "\n",exc_info=True)
        else:
            self.logger.error("\nERROR: " + msg + "\n",exc_info=True,)
    
    # if self.verbose:
    #    traceback.print_exc()
    
    def debug(self,msg):
        self.logger.debug(self.list2str(msg),exc_info=True)
    
    def exception(self,msg,*args,**kwargs):
        self.logger.exception(msg,*args,**kwargs)



class bcolors():
  """
  cls for printing in colors
  https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
  """
  def __init__ (self):
      super(bcolors, self).__init__()
      HEADER = '\033[95m'
      OKBLUE = '\033[94m'
      OKGREEN = '\033[92m'
      WARNING = '\033[93m'
      ERROR = '\033[91m'
      ENDC = '\033[0m'
      BOLD = '\033[1m'
      UNDERLINE = '\033[4m'
'''
