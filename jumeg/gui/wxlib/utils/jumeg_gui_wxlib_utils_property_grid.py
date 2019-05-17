#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 07.12.18
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import wx
from   wx.lib.scrolledpanel import ScrolledPanel
import wx.propgrid as wxpg
from   wx.propgrid import PropertyGridManager as wxpgm
from   pubsub      import pub

__version__='2019.05-14.001'

class JuMEG_wxPropertyGridManagerPanel(ScrolledPanel):
    """
    sub cls  Scrolled PropertyGrigManager with JuMEG_wxPropertyGrid_EditorDirDLG

    Example:
    --------
    from jumeg.gui.wxlib.utils.jumeg_gui_wxlib_utils_property_grid import JuMEG_wxPropertyGridManagerPanel,JuMEG_wxPropertyGrid_EditorDirDLG

    pgm_panel = JuMEG_wxPropertyGridManagerPanel(parent_panel,name="EXP_TMP_PROPERTY_GRID")
    pgm_panel.Clear()
   #--- PropertygridManager
    pgm = pgm_panel.PGM
   #--- add new Property Page to PropertyGridManager
    pgp = pgm.AddPage("TEST1")

    pg_stages=pgp.Append(wxpg.ArrayStringProperty(label="stages",name="STAGES",value=["/data1","/data2"]))
    #--- use user defined editor e.g. JuMEG_wxPropertyGrid_EditorDirDLG; needs registration before!!!
    pgp.SetPropertyEditor(""STAGES","JuMEG_wxPropertyGrid_EditorDirDLG")

    """
    
    def __init__(self,parent,name="PROPERTY_GRID_MANAGER_PANEL",**kwargs):
        super().__init__(parent,-1,name=name,style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER)
        self._property_grid_manager = None
        self._isRegisterd = False
        self._wx_init(**kwargs)
        self.ApplyLayout()
    
    @property
    def PropertyGridManager(self): return self._property_grid_manager
    
    @property
    def PGM(self): return self._property_grid_manager
    
    @property
    def PGMGrid(self): return self._property_grid_manager.GetGrid()
    
    #@property
    #def PGMTB(self): return self._property_grid_manager.GetToolBar()
    
    @property
    def isRegisterd(self): return self._isRegisterd
    
    def _register(self):
        """ register user PropertyGrids
            if not self._isRegisterd:
             # self.PGM.RegisterEditor(JuMEG_wxPropertyGrid_EditorDirDLG)
            self._isRegisterd = True
        """
        pass
        
    def _wx_init(self,**kwargs):
        style = wxpg.PG_BOLD_MODIFIED | wxpg.PGMAN_DEFAULT_STYLE | wx.propgrid.PG_TOOLTIPS  # wxpg.PG_DESCRIPTION
        self._property_grid_manager = wxpgm(self,-1,style=style)
        self._property_grid_manager.SetExtraStyle(wxpg.PG_EX_HELP_AS_TOOLTIPS)
        self._register()
        
    def Clear(self):
        self.PGM.Clear()
    
    def ApplyLayout(self):
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.Sizer.Add(self._property_grid_manager,1,wx.ALIGN_LEFT | wx.EXPAND | wx.ALL,1)
        self.SetSizer(self.Sizer)
        self.Sizer.Fit(self)
        self.SetAutoLayout(1)
        self.SetupScrolling()

class JuMEG_wxPropertyGridPageBase(JuMEG_wxPropertyGridManagerPanel):
   """
     PropertyGridManager Page Base CLS
     use for Template parameter keys (experiment,bids,fif_export ..)
   """
   def __init__(self,parent,**kwargs):
       super().__init__(parent,**kwargs)
       self._init(**kwargs)
       
   @property
   def prefix(self): return self._prefix + self._label.upper()

   @property
   def data(self):   return self._data

   def GetData(self):
       return  self.PGM.GetPropertyValues( as_strings=False,inc_attributes=False)

   def _init(self,**kwargs):
       self._data = {}
       self._default_update_from_kwargs(**kwargs)
       self.PGMPage = self.PGM.AddPage(label=self._label)
       self._update_from_kwargs(**kwargs)
       self.update(**kwargs)
       
   def _default_update_from_kwargs(self,**kwargs):
       self._label  = kwargs.get("label","TEST").capitalize()
       self._prefix = kwargs.get("prefix","PGM").upper()
       self._data   = kwargs.get("data",self._data)

   def _update_from_kwargs(self,**kwargs):
       pass
   
   def update(self,**kwargs):
       """ """
       pass

   def _set_property_ctrl(self,k,v,page=None):
        """
        :param prefix: Any,
        :param k: key,
        :param v: value
        """
        if not page:
           page = self.PGMPage
       
        if isinstance(v,(list)):
           page.Append(wxpg.ArrayStringProperty(k,name=k,value=v))
    
        elif isinstance(v,(bool)):
           page.Append(wxpg.BoolProperty(label=k,name=k,value=v))
           page.SetPropertyAttribute(k,"UseCheckbox",True)
        else:
           page.Append(wxpg.StringProperty(label=k,name=k,value=str(v) ))

class JuMEG_wxPropertyGridPageNotebookBase(wx.Notebook):
    """
    Base CLS
    show template properties within a wx.PropertyGridManager
    experment parameter
    
    
    :param prefix :  CLS prefix to find ctrl by GetName()
    :param title  : title
    :param data   : dict;  e.g. :jumeg-experiment-template <experiment> key,values
    :return
     wx.Propertygrid.Page
    
    :Example:
    ---------
    
     self.PropertyGridNoteBoook = JuMEG_wxTmpPGNB_Experiment(self.PanelA.Panel,name=self.title.replace(" ","_").upper() + "_TMP_PROP_GRID_NB")
     self.PropertyGridNoteBoook.update(data=data)
    """
    
    def __init__(self,parent,name="PGNB",**kwargs):
        super().__init__(parent,name=name)
        self._init(**kwargs)

    def _update_from_kwargs(self,**kwargs):
        self._prefix= kwargs.get("prefix","JUMEG_PGMP_NB_")
        self.title  = kwargs.get("title",self.GetName())
        if kwargs.get("data"):
           self._data.clear()
           self._data = kwargs.get("data")
        if kwargs.get("types"):
           self._types=kwargs.get("types",[])
     
    def _init(self,**kwargs):
        self._data   = { }
        self._pgmp   = { }
        self._pgp    = { }
        self._update_from_kwargs(**kwargs)
        self.DeleteAllPages()
        #self.update(**kwargs)

    def update(self,**kwargs):
        pass
        
    def GetData(self):
        """
        :return:
        """
        for type in self._types:
            if self._data.get(type):
               self._data[type].clear()
               self._data[type] = self._pgmp[type].GetData()
        return self._data


class JuMEG_wxPropertyGridSubProperty(wxpg.PGProperty):
    """ Demonstrates a property with few children.
    wxdemo  class SizeProperty(wxpg.PGProperty) line 123
    """
    def __init__(self, label, name = wxpg.PG_LABEL, value={"mne":"mne1","eeg":"eeg2"}):
        wxpg.PGProperty.__init__(self, label, name)
    
        self._key_list = [*value]
        for k in self._key_list:
            self.AddPrivateChild( wxpg.StringProperty(k, value=value.get(k) ))
        self.m_value = value
        
    def GetClassName(self):
        return self.__class__.__name__

    def DoGetEditorClass(self):
        return wxpg.PropertyGridInterface.GetEditorByName("TextCtrl")

    def RefreshChildren(self):
        v = self.m_value
        i = 0
        for k in self._key_list:
            self.Item(i).SetValue( v.get(k) )
            i+=1
   
    def ChildChanged(self, thisValue, childIndex, childValue):
        mval = self.m_value
        k    = self._key_list[childIndex]
        mval[k]=childValue
       
        return mval #self.m_value


        
'''
class PyObjectPropertyValue:
    """\
    Value type of our sample PyObjectProperty. We keep a simple dash-delimited
    list of string given as argument to constructor.
    """
    def __init__(self, s=None):
        try:
            self.ls = [a.strip() for a in s.split('-')]
        except:
            self.ls = []

    def __repr__(self):
        return ' - '.join(self.ls)


class PyObjectProperty(wxpg.PGProperty):
    """\
    Another simple example. This time our value is a PyObject.

    NOTE: We can't return an arbitrary python object in DoGetValue. It cannot
          be a simple type such as int, bool, double, or string, nor an array
          or wxObject based. Dictionary, None, or any user-specified Python
          class is allowed.
    """
    def __init__(self, label, name = wxpg.PG_LABEL, value=None):
        wxpg.PGProperty.__init__(self, label, name)
        self.SetValue(value)

    def GetClassName(self):
        return self.__class__.__name__

    def GetEditor(self):
        return "TextCtrl"

    def ValueToString(self, value, flags):
        return repr(value)

    def StringToValue(self, s, flags):
        """ If failed, return False or (False, None). If success, return tuple
            (True, newValue).
        """
        v = PyObjectPropertyValue(s)
        return (True, v)


class DictProperty(wxpg.PGProperty):
   """ Demonstrates a property with few children. """

   def __init__(self,label,name=wxpg.PG_LABEL,value={ }):
       wxpg.PGProperty.__init__(self,label,name)
    
       if value:
          self._keys = value.keys()
          print(self._keys)
    
       #value = self._ConvertValue(value)
       for k in self._keys:
           self.AddPrivateChild(wxpg.StringProperty(k,value=value[k]))
    
       self.m_value = value


   def GetClassName(self):
       return self.__class__.__name__
    
    
   def DoGetEditorClass(self):
       return wxpg.PropertyGridInterface.GetEditorByName("TextCtrl")
    
    
   def RefreshChildren(self):
        #size = self.m_value
        #self.Item(0).SetValue( size.x )
        #self.Item(1).SetValue( size.y )
       pass
    
    
   def _ConvertValue(self,value):
      """ Utility convert arbitrary value to a real wx.Size.
      """
      import collections
      if isinstance(value,collections.Sequence) or hasattr(value,'__getitem__'):
         value = wx.Size(*value)
      return value
    
    
   def _ChildChanged(self,thisValue,childIndex,childValue):
       #size = self._ConvertValue(self.m_value)
       #if childIndex == 0:
       #    size.x = childValue
       #elif childIndex == 1:
       #    size.y = childValue
       #else:
       #    raise AssertionError
       
       #return size
       pass

class JuMEG_wxPropertyGrid_EditorDirDLG(wxpg.PGTextCtrlEditor):
    """ https://wxpython.org/Phoenix/docs/html/wx.propgrid.PGMultiButton.html
       
       wxdemo -> More Windows7Controls ->propertyGrid
              -> class class DirsProperty line 167
              -> class SampleMultiButtonEditor line 312
              and register stuff:
              -> class TestPanel line 694
        
        Register editor class - needs only to be called once
        multiButtonEditor = SampleMultiButtonEditor()
        wx.propgrid.PropertyGrid.RegisterEditorClass(multiButtonEditor)
     
        ->Insert the property that will have multiple buttons
          propGrid.Append( wx.propgrid.LongStringProperty("MultipleButtons", wx.propgrid.PG_LABEL))
       ->Change property to use editor created in the previous code segment
         propGrid.SetPropertyEditor("MultipleButtons", multiButtonEditor)
     
     :example
     --------
     pgm = JuMEG_wxPropertyGridManagerPanel()
     pgm.register() => call to self.PGM.RegisterEditor(JuMEG_wxPropertyGrid_EditorDirDLG)
     
     
    """
    #def __init__(self, label="TEST", name = wxpg.PG_LABEL, value=[]):
    #    super().__init__(self, label, name, value)
    #    self.m_display = ''
    
    def __init__(self):
        super().__init__()
        
    #def GetName(self):
    #    return "SampleMultiButtonEditor"
    
    def _GenerateValueAsString(self, delim=None):
        """ This function creates a cached version of displayed text
            (self.m_display).
        """
        if not delim:
            delim = self.GetAttribute("Delimiter")
            if not delim:
                delim = ','

        ls = self.GetValue()
        if delim == '"' or delim == "'":
            text = ' '.join(['%s%s%s'%(delim,a,delim) for a in ls])
        else:
            text = ', '.join(ls)
        self.m_display = text


    def _StringToValue(self, text, argFlags):
        """ If failed, return False or (False, None). If success, return tuple
            (True, newValue).
        """
        delim = self.GetAttribute("Delimiter")
        if delim == '"' or delim == "'":
            # Proper way to call same method from super class
            return super(DirsProperty, self).StringToValue(text, 0)
        v = [a.strip() for a in text.split(delim)]
        return (True, v)

    
    
    def _dir_dlg(self,propgrid):
        dlg = wx.DirDialog(propgrid,message="Select a directory to be added to the list:",style= wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        retval = False
        if dlg.ShowModal() == wx.ID_OK:
           new_path = dlg.GetPath()
           old_value = None #self.m_value
           if old_value:
              new_value = list(old_value)
              new_value.append(new_path)
           else:
              new_value = [new_path]
              self.SetValueInEvent(new_value)
           retval = new_val
        else:
           dlg.Destroy()
        return retval
    
    def CreateControls(self, propGrid, property, pos, size):
       #--- Create and populate buttons-subwindow
        self.buttons = wxpg.PGMultiButton(propGrid, size)
       #--- Add two regular buttons
       
       
      # TODO
   # check for  button event
    
        self.buttons.Add("...")
       #--- Add a Bitmap Folder button for DirectoryDialog
        self.buttons.AddBitmapButton(wx.ArtProvider.GetBitmap(wx.ART_FOLDER))
        wnd = super().CreateControls(propGrid,property,pos,self.buttons.GetPrimarySize())
        wnd = wnd.m_primary
        self.buttons.Finalize(propGrid, pos)
        return wxpg.PGWindowList(wnd,self.buttons)
    
       
    def OnEvent(self, propGrid, prop, ctrl, evt):
        if evt.GetEventType() == wx.wxEVT_COMMAND_BUTTON_CLICKED:
           print("EVT Editor: ")
           obj = evt.GetEventObject()
           #print(obj.GetName())
         
           evtId = evt.GetId()
           print("EVT Editor: ".format(evtId))
           if evtId == self.buttons.GetButtonId(0):
           #--- show editor
              return True  # Return True  value changed
           if evtId == self.buttons.GetButtonId(1):
              return True #self._dir_dlg(propGrid)  # Return false since value did not change
               
        return super().OnEvent(propGrid, prop, ctrl, evt)
  
  
    

class DirsProperty(wxpg.ArrayStringProperty):
    """ Sample of a custom custom ArrayStringProperty.

        Because currently some of the C++ helpers from wxArrayStringProperty
        and wxProperytGrid are not available, our implementation has to quite
        a bit 'manually'. Which is not too bad since Python has excellent
        string and list manipulation facilities.
    """
    def __init__(self, label, name = wxpg.PG_LABEL, value=[]):
        wxpg.ArrayStringProperty.__init__(self, label, name, value)
        self.m_display = ''
        # Set default delimiter
        self.SetAttribute("Delimiter", ',')


    # NOTE: In the Classic version of the propgrid classes, all of the wrapped
    # property classes override DoGetEditorClass so it calls GetEditor and
    # looks up the class using that name, and hides DoGetEditorClass from the
    # usable API. Jumping through those hoops is no longer needed in Phoenix
    # as Phoenix allows overriding all necessary virtual methods without
    # special support in the wrapper code, so we just need to override
    # DoGetEditorClass here instead.
    def DoGetEditorClass(self):
        return wxpg.PropertyGridInterface.GetEditorByName("JuMEG_wxPropertyGrid_EditorDirDLG")


    def ValueToString(self, value, flags):
        # let's just use the cached display value
        return self.m_display


    def OnSetValue(self):
        self.GenerateValueAsString()


    def DoSetAttribute(self, name, value):
        retval = super(DirsProperty, self).DoSetAttribute(name, value)

        # Must re-generate cached string when delimiter changes
        if name == "Delimiter":
            self.GenerateValueAsString(delim=value)

        return retval


    def GenerateValueAsString(self, delim=None):
        """ This function creates a cached version of displayed text
            (self.m_display).
        """
        if not delim:
            delim = self.GetAttribute("Delimiter")
            if not delim:
                delim = ','

        ls = self.GetValue()
        if delim == '"' or delim == "'":
            text = ' '.join(['%s%s%s'%(delim,a,delim) for a in ls])
        else:
            text = ', '.join(ls)
        self.m_display = text


    def StringToValue(self, text, argFlags):
        """ If failed, return False or (False, None). If success, return tuple
            (True, newValue).
        """
        delim = self.GetAttribute("Delimiter")
        if delim == '"' or delim == "'":
            # Proper way to call same method from super class
            return super(DirsProperty, self).StringToValue(text, 0)
        v = [a.strip() for a in text.split(delim)]
        return (True, v)

    def _dir_dlg(self,propgrid):
        retval = False
        dlg = wx.DirDialog(propgrid,message="Select a directory to be added to the list:",style= wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
           new_path = dlg.GetPath()
           old_value = self.m_value
           if old_value:
              new_value = list(old_value)
              new_value.append(new_path)
           else:
              new_value = [new_path]
              self.SetValueInEvent(new_value)
              retval = True
          
           dlg.Destroy()
           return retval


    def __OnEvent(self, propgrid, primaryEditor, evt):
        if evt.GetEventType() == wx.wxEVT_COMMAND_BUTTON_CLICKED:
           obj = evt.GetEventObject()
           print(obj)
           evtId = obj.GetId()
           print("EVT ARRAY STRING: ".format(evtId))
          
            # self. _dir_dlg(self,propgrid)
           print("EVT Editor: ".format(evtId))
           if evtId == self.buttons.GetButtonId(0):
         #--- show editor
              return True  # Return True  value changed
           if evtId == self.buttons.GetButtonId(1):
              return  self._dir_dlg(propGrid)  # Return false since value did not change

           return super().OnEvent(propGrid,prop,ctrl,evt)
'''