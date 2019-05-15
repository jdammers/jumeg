#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:16:46 2018

@author: fboers
-------------------------------------------------------------------------------
ToDo:
implement support for ssh to pc & ssh cluster via PBS protocoll
-------------------------------------------------------------------------------  
https://pythonspot.com/python-subprocess/


for wx log
https://www.blog.pythonlibrary.org/2009/01/01/wxpython-redirecting-stdout-stderr/

"""

import sys,os,shlex,re,wx,io
import threading

from datetime         import datetime as dt
from subprocess       import Popen, PIPE,STDOUT
from pubsub           import pub
from jumeg.base.jumeg_base import jumeg_base as jb
from jumeg.gui.wxlib.jumeg_gui_wxlib_pbshost  import JuMEG_PBSHostsParameter

import logging
logger = logging.getLogger('jumeg')

__version__="2019.05.14.001"


'''
https://gist.github.com/bortzmeyer/1284249
HOST="www.example.org"
# Ports are handled in ~/.ssh/config since we use OpenSSH
COMMAND="uname -a"

ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
                       shell=False,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
result = ssh.stdout.readlines()
if result == []:
    error = ssh.stderr.readlines()
    print >>sys.stderr, "ERROR: %s" % error
else:
print result

'''

class JuMEG_IoUtils_SubProc_Base(object):
   #__slots__ =["proc","args","cmd","use_shell","stdout","stdin"]
   def __init__( self, **kwargs ):
       super().__init__()
       self._proc     = None
       self._pid      = None
       self._args     = None
       self._cmd      = None
       self._use_shell= True
       self._stdout   = None
       self._stderr   = None
   
   @property
   def job_number(self): return self._jobnumber
   #---
   @property
   def cmd(self): return self._cmd
   #---
   @property
   def args(self): return self._args
   #---
   @property
   def proc(self): return self.__proc
 
class JuMEG_IoUtils_SubProc_LogMsg(object):
   """
   ToDO: update logger
   """
   def __init__( self, **kwargs ):
       super().__init__()
       self._evt_msg_key = {"error" : "MAIN_FRAME.MSG.ERROR",
                            "isbusy": "MAIN_FRAME.STATUS.BUSY",
                            "stb"   : "MAIN_FRAME.STB.MSG"}
       self._init(**kwargs)
       
   def GetEvtMSGKey(self,key): return self._evt_msg_key[key]
   
   @property
   def msg_prefix(self):
       s=""
       if self.name:
          s = " --> {}".format(self.name)
       if self.hostname:
          s+= "@{}".format(self.hostname)
       if self.id:
          s+= "-{:03}:".format(self.id)
       return s
   
   def _update_from_kwargs(self,**kwargs):
       self.name     = kwargs.get("name","RunOnLocal")
       self.hostname = kwargs.get("hostname","local")
       self.id       = kwargs.get("id",0)
       self.verbose  = kwargs.get("verbose",False)
       self.pubsub_listener_stb = kwargs.get("pubsub_listener_stb","MAIN_FRAME.STB.MSG")
       
   def _init(self,**kwargs ):
       self._update_from_kwargs(**kwargs)
       
   def msg_start(self):
       wx.LogMessage("{} START\n".format(self.msg_prefix))

   def msg_stop(self):
       wx.LogMessage("{} STOP\n".format(self.msg_prefix))

   def msg_cmd(self,cmd):
       wx.LogMessage(jb.pp_list2str(cmd,head=self.msg_prefix + " cmd:"))
   
   def msg(self,msg,head=""):
       if isinstance(msg,(list,dict)):
          wx.LogMessage(jb.pp_list2str(msg,head=self.msg_prefix +" "+head +"\n"))
       else:
          wx.LogMessage(self.msg_prefix +" "+head +"\n"+msg)
  
   def msg_job_pid(self,jobinfo,pid):
       wx.LogMessage("{} RUN SubProc Nr.: {}  PID: {}\n".format(self.msg_prefix,jobinfo,pid))
       self.post_event(self.pubsub_listener_stb,data=["RUN", "Job: " +str(jobinfo), "PID", str(pid)])
 
   def msg_done(self,jobnr=None,pid=None):
       wx.LogMessage("{} DONE  Job Nr: {} PID: {}\n".format(self.msg_prefix,jobnr,pid))
       self.post_event(self.pubsub_listener_stb,data=["DONE", "", "", ""])

   def msg_stdout(self,msg):
       if msg:
          if isinstance(msg,(list)):
             msg="\n".join(msg)
          elif not jb.isNotEmptyString(msg):
             s = str(msg,'utf-8') # byte blob
             wx.LogMessage(self.msg_prefix +"\n" + re.sub(r'\n+',"\n",s).strip())
          else:
              wx.LogMessage(self.msg_prefix+"\n"+msg)
       else:
          wx.LogMessage(self.msg_prefix + " STDOUT: None")
    
   def msg_stderr(self,msg,verbose=False):
       """
       ToDo send pubsub MSG to count Error status + errormsg
       :param msg:
       :return:
       """
       if msg:
          if isinstance(msg,(list)):
             msg="\n".join(msg)
          elif not jb.isNotEmptyString(msg):
             s = str(msg,'utf-8')  # byte blob
             wx.LogSysError(self.msg_prefix + " STDERR:\n" + re.sub(r'\n+',"\n",s).strip())
          else:
             wx.LogSysError(self.msg_prefix + " STDERR:\n" + msg)
       elif verbose:
          wx.LogSysError(self.msg_prefix + " STDERR: None")
    
   def post_event(self,message,data):
       wx.CallAfter(lambda *a:pub.sendMessage(message,data=data))

   def msg_split_and_show(self,msg_std):
       """
       split input to log messages stdout and stderr
       .:param string with stdout and stderr
       
       """
       if not msg_std : return
       lines = msg_std.split("\n")
       
       while (len(lines)):
           #--- msg no error
             if (lines[0].startswith("JuMEG LOG") and (lines[0].find("ERROR") < 1)):
                msg=lines.pop(0) +"\n"
                while (len(lines)):
                    if lines[0].startswith("JuMEG LOG"):
                       break
                    else:
                       msg += lines.pop(0) + "\n"
                self.msg_stdout(msg)

             elif (lines[0].startswith("JuMEG LOG") and (lines[0].find("ERROR") > 1)):
                  msg =lines.pop(0)+"\n"
                  while (len(lines)):
                      if lines[0].startswith("JuMEG LOG"):
                         break
                      else:
                         msg += lines.pop(0)+"\n"
                  self.msg_stderr(msg)
             else:
                 lines.pop(0)

class JuMEG_IoUtils_SubProcThreadBase(threading.Thread):
    """
    base cls
    https://www.python-kurs.eu/threads.php
    https://www.oreilly.com/library/view/python-cookbook/0596001673/ch06s03.html
    
    ToDo send pubsub MSG to count finished status + errormsg
    """
    def __init__( self, **kwargs ):
        super().__init__()
        self._stopevent = threading.Event(  )
     
      #-- proc
        self._proc      = None
        self._args      = None
        self._cmd       = None
        self._stdout    = None
        self._stderr    = None
        self.use_shell  = True
        
        self._isbusy    = False
        
        self._id        = 0
        self.name       = "RunOnLocal"
        self.joblist    = None
        self.verbose    = False
       #---
        self.Host       = JuMEG_PBSHostsParameter()
        self.Log        = JuMEG_IoUtils_SubProc_LogMsg()
       
       #--- call _init in sub class
       # self._init(**kwargs)

    @property
    def id(self): return self._id
    @id.setter
    def id(self,v):
        self._id = v
        if self.Log: self.Log.id=v
    
    @property
    def isBusy(self): return self.__isbusy
    
    def check_joblist(self):
        """
        ck if joblist is not none,
        post event msg if None
        or converts to list if needed
        
        :return True/False
        
        todo raise exception if None
        """
        if not self.joblist:
           
           s= "< "+jb.get_function_name() +" > command or joblist not defined\n ---> in module:\n ---> "+ jb.get_function_fullfilename()
           self.Log.post_event( self.Log. GetEvtMSGKey("error"),data=s)
           return
        
        if not isinstance(self.joblist, list):
           self.joblist = [self.joblist]

        return True
        
    def _set_isbusy(self,status):
        self.__isbusy = status
        self.Log.post_event("isbusy",data=status)

    def check_prefix_and_python_version(self):
        cmd=[]
        if self.Host.cmd_prefix:
           cmd.append(self.Host.cmd_prefix)
        if self.Host.python_version.startswith("python"):
           cmd.append(self.Host.python_version)
        if cmd:
           return (" ".join(cmd)).strip()
        return ""
   
    def _update_from_kwargs(self,**kwargs):
        self.joblist = kwargs.get("jobs",self.joblist)
        self.name    = kwargs.get("name",self.name)
        self.verbose = kwargs.get("verbose",self.verbose)
        self.id      = kwargs.get("id",self.id)
        if kwargs.get("host_parameter"):
           self.Host.update(**kwargs.get("host_parameter"))

    def _init(self,**kwargs ):
        self._update_from_kwargs(**kwargs)
        self.update(**kwargs)
        self.update_log()
        
    def update_log(self):
        self.Log.name = self.name
        if self.Host:
           self.Log.hostname = self.Host.name
        
    def update(self,**kwargs):
        """ overwrite"""
        pass
    
    def run(self):
        """Run Worker Thread overwrite"""
        pass
    
    def join(self, timeout=None):
        """
        Stop the thread.
        https://www.oreilly.com/library/view/python-cookbook/0596001673/ch06s03.html
        """
        self._stopevent.set(  )
        threading.Thread.join(self, timeout)


class JuMEG_IoUtils_SubProcThreadLocal(JuMEG_IoUtils_SubProcThreadBase):
   """
   runs on local machine in a thread with Popen-call
   """
   def __init__(self,**kwargs):
       super().__init__()
       self._init(**kwargs)
       
   def run(self):
       """
       execute command on local host

       Parameter
       ---------
       cmd: string <None>
       """
       self._pidlist = []
       self._job_number = 0
      
       if self.verbose:
          self.Log.msg(self.joblist, head="Thrd RunOnLocal Job list:")
          self.Log.msg(self.Host.get_parameter(),head="Parameter" )
       
       if not self.check_joblist(): return
       self._set_isbusy(True)
   
       for job in self.joblist:
           if isinstance(job, (list)):
              cmd = " ".join(job)
           else:
              cmd = job

            #   self._args = shlex.split(job)
            
           cmd  = " ".join( [self.check_prefix_and_python_version(),cmd ]).strip()
           
           if self.verbose:
              self.Log.msg(cmd,"Thrd Process")
              self.Log.msg_start()

           self.__proc = Popen(cmd,stdout=PIPE,stderr=STDOUT,shell=self.use_shell,universal_newlines=True)
           self._pidlist.append(self.__proc.pid)
           self._job_number+=1
           self.Log.msg_job_pid( str(self._job_number)+"/"+str(len(self.joblist)),self.__proc.pid)
        
           (msg_stdout,_) = self.__proc.communicate()  # BUG ??? returns only one output mixed with stdout and stderr

           if self.verbose:
              self.Log.msg_stop()
              self.Log.msg_split_and_show(msg_stdout)
              
           if self._stopevent.isSet(): break
           
       self._set_isbusy(False)
       self.Log.msg_done(jobnr=self._job_number,pid=self.__proc.pid)

class JuMEG_IoUtils_SubProcThreadPBS(JuMEG_IoUtils_SubProcThreadBase):
   """
   runs on a remote machine in a PBS environoment
   e.g. cluster system
   """
   def __init__( self, **kwargs ):
       super().__init__()

class JuMEG_IoUtils_SubProcThreadSSH(JuMEG_IoUtils_SubProcThreadBase):
   """
   runs on a remote machine in a PBS environoment
   e.g. cluster system
   """
   def __init__( self, **kwargs ):
       super().__init__()


class JuMEG_IoUtils_SubProcess(object):
    """
    jumeg subclass for subprocess module
    
    run on local machine
    
    ToDo
    -----
    implement support for ssh to pc & ssh cluster via PBS protocol
    """
    def __init__(self,**kwargs):
        super().__init__()

      # self.hostinfo_default={'kernels': 1, 'maxkernels': 1, 'maxnodes': 1, 'name': 'local', 'nodes': 1,'python_version':"python3"},
      # self.hostinfo = self._hostinfo_default
        #self.Thrd = JuMEG_IoUtils_SubProcThread()

        self._args      = None
        self._cmd       = None

        self._stdout    = None
        self._stderr    = None
        self.__proc     = None
        self.__isBusy   = False
        self._thrd_list = []

        #self._init(**kwargs)

        self._init_pubsub_messages()


    #def _init(self,**kwargs):
    #    self.verbose = kwargs.get("verbose", False)
        #self.Host.update( kwargs.get("host", None) )
   #---
    @property
    def cmd(self): return self._cmd
   #---
    @property
    def args(self): return self._args
   #---
    @property
    def proc(self): return self.__proc
   #---   
    @property
    def isBusy(self): return self.__isBusy

    def _init_pubsub_messages(self):
        pub.subscribe(self.run,'SUBPROCESS.RUN.START')
        pub.subscribe(self.stop,'SUBPROCESS.RUN.STOP')

        #pub.subscribe(self.run_on_remote, 'SUBPROCESS.RUN.REMOTE')
        #pub.subscribe(self.run_on_cluster,'SUBPROCESS.RUN.CLUSTER')
        pass

    def stop(self,id):
        """
        
        ToDo stop thrd
        find thrd via pid
        if local
           send stop signal and join th thrd,
            kill via sys signals
            wait till terminate
        kill via ssh
        ToDo
        check  delete trhd obj for memory?
        """
        if int(pid):
           self._thrd_list[id].join()
        
    
    def cancel(self,id=None):
        self.stop(id)
        
    def run(self,jobs=None,host_parameter=None,verbose=False):
        """
        https://wiki.wxpython.org/LongRunningTasks

        :param joblist:
        :param hostobj:
        :param verbose
        :return:
        """
        if self.isBusy: return
        if not host_parameter: return
        self.__isBusy=True
        
        if host_parameter.get("name") == "local":
           thrd = JuMEG_IoUtils_SubProcThreadLocal(jobs=jobs,host_parameter=host_parameter,verbose=verbose)
        elif host_parameter.name.endswith("cluster"):
           thrd = JuMEG_IoUtils_SubProcThreadPBS(jobs=jobs,host_parameter=host_parameter,verbose=verbose)
        else:
           thrd = JuMEG_IoUtils_SubProcThreadSSH(jobs=jobs,host_parameter=host_parameter,verbose=verbose)

        self._thrd_list.append(thrd)
       #--- ToDo delete thrd from list vid idx id
        thrd.id = len(self._thrd_list)
        #thrd.update(joblist=joblist,hostobj=hostobj,verbose=verbose)
        
        thrd.start()

        self.__isBusy=False
        
        return thrd.id


class JuMEG_IoUtils_SubProcThread(JuMEG_IoUtils_SubProcThreadBase):
    """SubProc Worker Thread Class."""

   # ----------------------------------------------------------------------
    def __init__(self,**kwargs):
        """
        hostinfo: dict with keys of Host CLS
        joblist : None the jobs/cmds to execute
        verbose : False
        """
        super().__init__()
        self.Host = JuMEG_PBSHostsParameter(kernels=1, maxkernels=1, maxnodes=1, name="local", nodes=1,python_version="python3")

        self._cmd_prefix = "/usr/bin/env"

        self._jobnumber = 0
        self._args    = None
        self._cmd     = None
        self._use_shell=True
        self._stdout  = None
        self._stderr  = None
        self.__proc   = None
        self._init(**kwargs)

    def _init( self, **kwargs ):
        self._update_from_kwargs(**kwargs)

    @property
    def job_number(self): return self._jobnumber
   #---
    @property
    def cmd(self): return self._cmd
   #---
    @property
    def args(self): return self._args
   #---
    @property
    def proc(self): return self.__proc
   #---
    def _update_from_kwargs(self,**kwargs):
        self.joblist = kwargs.get("joblist","ls -lisa")
        self.verbose = kwargs.get("verbose", False)
        if kwargs.get("hostinfo"):
           self.Host.update(**kwargs.get("hostinfo"))

    def post_event(self,message,data):
        wx.CallAfter(lambda *a: pub.sendMessage(message,data=data))

   # ----------------------------------------------------------------------
    def run(self): #,**kwargs):
        """Run Worker Thread."""
        if not self.joblist: return

        if self.Host.name == "local":
           self.run_on_local()
        else:
           self.run_on_cluster()

   # ----------------------------------------------------------------------
    def run_on_local(self):
        """
         execute command on local host

         Parameter
         ---------
          cmd: string <None>
        """
        self._pidlist = []

        if not self.joblist:
           self.post_event("MAIN_FRAME.MSG.ERROR",data="JuMEG_IoUtils_SubProcess.run_on_local: no command defined")
           return

        self.post_event("MAIN_FRAME.STATUS.BUSY", data=True)

        self.__isBusy = True
        self._args = None

        if not isinstance(self.joblist, list):
           self.joblist = list(self.joblist)

        #if self.verbose:
        #   wx.LogMessage(jb.pp_list2str(self.joblist, head="PBS SubProc Job list: "))
        #   wx.LogMessage(jb.pp_list2str(self.host,    head="PBS SubProc HOST Info:"))

        self._job_number=0
        self.post_event("MAIN_FRAME.STATUS.BUSY", data=True)
        self.__isBusy = True

        for job in self.joblist:
            if isinstance(job, list):
               self._args = job
               self._cmd = " ".join(job)
            else:
               self._args = shlex.split(job)
               self._cmd = job

            if not self._args:
               self.post_event("MAIN_FRAME.MSG.ERROR",data="JuMEG_IoUtils_SubProcess.run_on_local error in <command>")
               return

            if self.verbose:
               self.log_info_process()
               self.log_info_start_process()

           # --- init & start process
            if self.Host.python_version.startswith("python"):
               self._args[0] = self._cmd_prefix +" "+self.Host.python_version+" "+self._args[0]

            arg_str= " ".join(self._args)
            wx.LogMessage("158 subporc "+arg_str )
            wx.LogMessage("159 subporc ".format( self.Host.get_parameter() ) )

            #with Popen(["ifconfig"], stdout=PIPE) as proc:
                 #log.write(proc.stdout.read())

            self.__proc = Popen(arg_str, stdout=PIPE, stderr=PIPE,shell=True)
            self._pidlist.append(self.__proc.pid)

            self._job_number+=1
            wx.LogMessage(" -->RUN SubProc Nr.: {}  PID: {}".format(self.job_number,self.__proc.pid))
            self.post_event("MAIN_FRAME.STB.MSG",data=["RUN", self._args[0], "PID", str(self.__proc.pid)])

            self._stdout, self._stderr = self.__proc.communicate()

            if self.verbose:
               self.log_info_stop_process()
               #self.log_info_stdout()
               #self.log_info_stderr()

        self.post_event("MAIN_FRAME.STATUS.BUSY",data=False)
        self.__isBusy = False

    def run_on_remote(self):  # ,host=None):
        """
        to do implement ssh and host list
        """
        pass

    def run_on_cluster(self):  # ,host=None):
        """
        to do implement ssh and cluster list
        cluster nodes and kernels
        """
        pass

    def log_info_start_process(self):
        wx.LogMessage("  -> start process Host: {}\n".format(self.Host.name))

    def log_info_stop_process(self):
        wx.LogMessage("  -> stop  process Host: {}\n".format(self.Host.name))

    def log_info_process(self):
        wx.LogMessage(jb.pp_list2str(self.cmd,head=" -->SubProcess cmd:"))

    def log_info_stdout(self):
        if self._stdout:
           s = str(self._stdout, 'utf-8')
           wx.LogMessage(" -->SubProcess output: " + re.sub(r'\n+', "\n", s).strip())
        else:
           wx.LogMessage(" -->SubProcess no output")

    def log_info_stderr(self):
        if self._stderr:
           s = str(self._stderr, 'utf-8')
           wx.LogError(" -->SubProcess Error: "+ re.sub(r'\n+',"\n" ,s).strip())
        else:
           wx.LogMessage(" -->SubProcess no ERROR")



        
''' 
import wx, sys
from threading import Thread
import time
import subprocess


class mywxframe(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self,None)
        pnl = wx.Panel(self)
        szr = wx.BoxSizer(wx.VERTICAL)
        pnl.SetSizer(szr)
        szr2 = self.sizer2(pnl)
        szr.Add(szr2, 1, wx.ALL|wx.EXPAND, 10)
        log = wx.TextCtrl(pnl, -1, style= wx.TE_MULTILINE, size = (300, -1))
        szr.Add(log, 0, wx.ALL, 10)
        btn3 = wx.Button(pnl, -1, "Stop")
        btn3.Bind(wx.EVT_BUTTON, self.OnStop)
        szr.Add(btn3, 0, wx.ALL, 10)
        self.CreateStatusBar()

        redir = RedirectText(log)
        sys.stdout=redir

        szr.Fit(self)
        self.Show()

    def sizer2(self, panel):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.tc2 = wx.TextCtrl(panel, -1, 'Set Range', size = (100, -1))
        btn2 = wx.Button(panel, -1, "OK",)
        self.Bind(wx.EVT_BUTTON, self.OnStart, btn2)
        sizer.Add(self.tc2, 0, wx.ALL, 10)
        sizer.Add(btn2, 0, wx.ALL, 10)
        return sizer


    def OnStart(self, event):
        self.p=subprocess.Popen(["C:\Python27\python.exe",'P:\Computing and networking\Python\Learning programs\hello_world.py'])

    def OnStop(self, event):
        self.p.terminate()

    def write(self, *args):
        print args

class RedirectText(object):
    def __init__(self, aWxTextCtrl):
        self.out=aWxTextCtrl

    def write(self, string):
        (self.out.WriteText, string)


app = wx.App()
frm = mywxframe()
app.MainLoop()


TextCtrl update by using wx.CallAfter

# spawn the new process, and redirecting stdout and stderr into this process
proc = subprocess.Popen([PathToCurrentPythonInterpreter, pyFilePath],stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1)
proc.stdin.close()
#create new stdout and stderr stream listener objects
m=StreamWatcher()
n=StreamWatcher()

#create new threads for each listener, when the listener hears something it prints it to the GUI console log area
proc1 = Thread(target=m.stdout_watcher, name='stdout-watcher', args=('STDOUT', proc.stdout))
proc1.daemon=True
proc1.start()
proc2 = Thread(target=n.stderr_watcher, name='stderr-watcher', args=('STDERR', proc.stderr))
proc2.daemon=True
proc2.start()



class StreamWatcher:
    def stdout_watcher(self, identifier, stream):
        #for line in stream:
        for line in iter(stream.readline,b''):
            print line
        if not stream.closed:
            stream.close()
        print "-i- Thread Terminating : ", identifier,"\n"

    def stderr_watcher(self, identifier, stream):
        #for line in stream:
        for line in iter(stream.readline,b''):
            #print (identifier, line)
            print "-e- %s" % line
        if not stream.closed:
            stream.close()
        print "-i- Thread Terminating : ", identifier, "\n"









HOST="127.0.0.1"
ssh = subprocess.Popen(["ssh",
                        "%s" % HOST],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        bufsize=0)
 
# send ssh commands to stdin
ssh.stdin.write("ls .\n")
ssh.stdin.write("uname -a\n")
ssh.stdin.write("uptime\n")
ssh.stdin.close()

# fetch output
for line in ssh.stdout:
    print(line),
'''


"""
# os.chdir()
cmd='jumeg_merge_meeg.py --eeg_stage=/home/fboers/MEGBoers/data/exp/MEG94T/eeg/MEG94T --meg_stage=/home/fboers/MEGBoers/data/exp/MEG94T/mne --eeg_filename=109887_MEG94T0T_01.vhdr --meg_filename=109887/MEG94T0T/130626_1329/1/109887_MEG94T0T_130626_1329_1_c,rfDC-raw.fif --eeg_response_shift=1000 --meg_stim_channel="STI 014" --bads_list="MEG 007,MEG 142,MEG 156,RFM 011" --eeg_stim_type=STIMULUS --startcode=128 --eeg_stim_channel="STI 014" -r -v'
args = shlex.split(cmd)
print(args)
p = subprocess.Popen(args)

print p
print "DONE"

from subprocess import Popen, PIPE

process = Popen(['swfdump', '/tmp/filename.swf', '-d'], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()


from subprocess import Popen, PIPE
 
process = Popen(['cat', 'test.py'], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()
print stdout

subprocess.call(args, *, stdin=None, stdout=None, stderr=None, shell=False)
# Run the command described by args. 
# Wait for command to complete, then return the returncode attribute.


 subprocess.check_output(args, *, stdin=None, stderr=None, shell=False, universal_newlines=False)
 # Run command with arguments and return its output as a byte string.
 
 s = subprocess.check_output(["echo", "Hello World!"])
print("s = " + s)



import subprocess
import time
import wx

from threading import Thread


class PingThread(Thread):

    def __init__(self, text_ctrl):
        Thread.__init__(self)
        self.text_ctrl = text_ctrl
        self.sentinel = True
        self.start()

    def run(self):            
        proc = subprocess.Popen("ping www.google.com",
                                     shell=True,
                                     stdout=subprocess.PIPE)
        while self.sentinel:
            line = proc.stdout.readline()
            if line.strip() == "":
                pass
            else:
                wx.CallAfter(self.text_ctrl.write, line)

            if not line: break

        proc.kill()


https://stackoverflow.com/questions/45289661/how-to-get-results-from-subprocess-stdout-and-display-them-realtime-in-textctrl
class MyFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title='Redirecter')
        self.ping_thread = None

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        panel = wx.Panel(self)

        self.log = wx.TextCtrl(panel, style=wx.TE_MULTILINE)

        ping_btn = wx.Button(panel, label='Ping')
        ping_btn.Bind(wx.EVT_BUTTON, self.on_ping)

        main_sizer.Add(self.log, 1, wx.ALL|wx.EXPAND, 5)
        main_sizer.Add(ping_btn, 0, wx.ALL, 5)
        panel.SetSizer(main_sizer)

        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.Show()

    def on_ping(self, event):
        self.ping_thread = PingThread(self.log)

    def on_close(self, event):
        if self.ping_thread:
            self.ping_thread.sentinel = False
            self.ping_thread.join()
        self.Destroy()


if __name__ == '__main__':
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()


"""