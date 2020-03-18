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


https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python

ToDo: check LogBook
https://github.com/getlogbook/logbook
'''

__version__="2019.05.14.001"


#===========================================================
#=== test logging stdout, stderr
#===========================================================
def test_log_std(txt):
  
   #--- log stdout,stderr
   jumeg_logger.log_stdout(label=" LOGTEST")
   jumeg_logger.log_stderr()
   
   print("  -> TEST print to logger: {}".format(txt) )

  #--- return back stdout/stderr from logger
   jumeg_logger.log_stdout(reset=True)
   jumeg_logger.log_stderr(reset=True)
   
   return raw_fname,raw

class StreamLogger(object):
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
    sys.stdout  = StreamLogger(logname="root",level=logging.INFO,label="TESTLOG OUT")
    sys.stderr = StreamLoggerS(logname="root",level=logging.INFO,label="TESTLOG ERR")
   
   """
   def __init__(self,logger=None,logname="jumeg",level=logging.INFO,label=None):
   #def __init__(self,logger=None,logname="jumeg",level=logging.INFO,label=None,logstdout=False,logstderr=False):
      
       """
       
       :param logger:  logger obj
       :param logname: logger obj name <jumeg>
       :param level:   <logging.INFO>
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
  
  # ToDo add as context manager using enter, exit
  # def __enter__(self):
  
  # def __exit__(selfself):
  #     sys.stdout = sys.__stdout__
   
   
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


class StreamLoggerSTD(object):
   """
   Example
   -------
    LogSTD = StreamLoggerSTD(logname="jumeg",level=logging.INFO,label=None)
    
    #--- log STDOUT and STDERR stream with jumeg_logger
     LogSTD.log_stdout()
     LogSTD.log_stderr()
   
    #--- unlog/reset stream
    
    LogSTD.unlog_stdout()
    LogSTD.unlog_stderr()
   
   """
   def __init__(self,**kwargs):
       self._log_stdout = True
       self._log_stderr = True
       self._is_stdout  = False
       self._is_stderr  = False

       self._update_kwargs(**kwargs)
   
   @property
   def IsSTDOUTlogged(self): return self._is_stdout

   @property
   def IsSTDERRlogged(self):
       return self._is_stderr

   def _update_kwargs(self,**kwargs):
       self._log_stdout = kwargs.get("logstdout",self._log_stdout)
       self._log_stderr = kwargs.get("logstderr",self._log_stderr)
       
   def log_stdout(self,**kwargs):
       sys.stdout = StreamLogger(**kwargs)
       self._is_stdout = True

   def unlog_stdout(self):
       sys.stdout = sys.__stdout__
       self._is_stdout = False

   def log_stderr(self,**kwargs):
       sys.stderr = StreamLogger(**kwargs)
       self._is_stderr =True
       
   def unlog_stderr(self):
       sys.stderr = sys.__stderr__
       self._is_stderr = False

  #--- contextmanager part  with block
   def __enter__(self,**kwargs):
       if self._log_stdout:
          self.log_stdout(**kwargs)
      
       if self._log_stderr:
          self.log_stderr(**kwargs)
       
   def __exit__(self,exc_type, exc_value, exc_traceback):
       """
       https://alysivji.github.io/managing-resources-with-context-managers-pythonic.html
       
       :param exc_type:
       :param exc_value:
       :param exc_traceback:
       :return:
       """
       self.unlog_stdout()
       self.unlog_stderr()


 
def log_stdout(reset=False,**kwargs):
    if reset:
       sys.stdout = sys.__stdout__
    else:
       sys.stdout = StreamLogger(**kwargs)
    return sys.stdout

def log_stderr(reset=False,**kwargs):
    if reset:
       sys.stderr = sys.__stderr__
    else:
       sys.stderr = StreamLogger(**kwargs)
    return sys.stderr


class JuMEGLogFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors
    https://docs.python.org/3/library/logging.html#logrecord-attributes
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
        logging.INFO: "%(levelname)s - %(asctime)s - %(name)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s\n",
        # logging.INFO:   "\n%(levelname)s - %(asctime)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s",
        logging.ERROR: "\n%(levelname)s - %(asctime)s - %(name)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s\n\n",
        logging.WARNING: "\n%(levelname)s - %(asctime)s - %(name)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s\n",
        logging.DEBUG: "%(levelname)s - %(asctime)s - %(name)s - %(module)s - %(funcName)s:%(lineno)d :\n%(message)s\n"
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
    
    def __init__(self,fname=None,prefix=None,name="jumeg_logger",postfix="001",extention=".log",path=None,level=None,mode="a"):
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
        
def setup_script_logging(fname=None,name=None,opt=None,level="DEBUG",logger=None,path=None,version=None,logfile=False,
                         mode="a",captureWarnings=True,logname="jumeg"):
    """
    setup logger :
     loging level, log handler, logs information e.g input parameter
    
    :param fname  : name of logfile
                    if fname is defined no <auto generated log filename> will be generated
    :param name   : name of script as part of <auto generated log filename> if logfile==True
    :param opt    : option obj from argpraser
                    opt, parser = get_args(argv)
    
    :param logger     : logger <logger obj> if <None> get new logger obj
    :param logname    : logger obj name <jumeg>
    :param level      : logging level <DEBUG>
    :param logfile    : flag log to logfile, will overwrite opt.logfile
    
    :param path       : logfile path <None> :
    :param mode       : logfile mode <a>  [a,w]  append or write
    :param version    : version  e.g.__version__
    :param captureWarnings: capture Warnings form wranings module <True>
    
    :return:
    logger
    
    """
    if not logger:
       logger=logging.getLogger(logname)
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

    
def update_filehandler(logger=None,logname="jumeg",**kwargs):
    """
    close all filehandlers first and add a new filehandler to the logger
    https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a
    
    :param logger  : the logger obj <None>
    :param logname : logger obj  name <jumeg>, used if logger is None
    
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
       logger=logging.getLogger(logname)
      
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
def getLogger(logname="jumeg",captureWarnings=True,level="INFO"):
    """
    get a logger, calls logging.getLogger(), add a Stream- and a FileHandler to the logger
    
    :param logname:  logger obj name <jumeg>
    :param level  :  log level <INFO>
                     [NOTSET,INFO,DEBUG,WARNINGS/WARN,ERROR,CRITICAL] or logging.XYZ or [0,10,20,30,40,50]
    :param captureWarnings: capture and log printed warning messages <True>
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
    
    logger = logging.getLogger(loger_name)
    logger.setLevel(level)
    logger.addHandler(LogStreamHandler())
    logger.addHandler(LogFileHandler())
    logging.captureWarnings(captureWarnings)
    
    return logger

