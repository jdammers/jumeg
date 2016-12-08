# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:05:40 2015

@author: fboers
"""

import wx
import wx.lib.colourdb as WX_CDB
from wx.combo import BitmapComboBox

#import wx.grid as gridlib
#---

class DLGButtonPanel(wx.Panel):
     
      def __init__(self,parent,color='grey90',buttons=None,*args,**kwargs):
          """Create the Button Panel."""
          bt_list=[(wx.ID_OK,self.OnClickOk,True)]
        # bt_list=[(wx.ID_CANCEL,self.OnClickOk,False),(wx.ID_OK,self.OnClickOk,True)]
          if not buttons:
                 buttons = bt_list
          
          wx.Panel.__init__(self, parent, *args, **kwargs)
          self.SetBackgroundColour(color)
          hbox = wx.BoxSizer(wx.HORIZONTAL)
          hbox.Add((-1,5),1,wx.EXPAND|wx.ALL)
          
          for bt in buttons:
              wbt = wx.Button(self,bt[0])
              wbt.Bind(wx.EVT_BUTTON,bt[1])
              if bt[2]:
                 wbt.SetDefault() 
              hbox.Add(wbt,0,wx.LEFT,border=5)
          self.SetSizer(hbox)
     
      def OnClickOk(self,evt):
          evt.Skip()
          #self.GetParent().close()          
          

########################################################################
class ColorComboBoxDialog(wx.Dialog):
    """"""

   # import wx.lib.colourdb as WX_CDB
    
 
    #----------------------------------------------------------------------
    def __init__(self,title="JuMEG TSV Groups Color Dialog",color_label=None,color_list=None):
        """Constructor"""
       # wx.Dialog.__init__(self,title=title) 
        super(ColorComboBoxDialog,self).__init__(None,title=title) 
        
    
        self.color_list_default = ['BLACK','RED','AQUAMARINE','BLUE','MEDIUMBLUE','MIDNIGHTBLUE','ROYALBLUE','NAVYBLUE','CYAN',
                                   'GREEN','DARKGREEN','YELLOW','MAGENTA','VIOLET','PURPLE1','GREY40','GREY50','GREY60','GREY70',
                                   'GOLD','PERU','BROWN','ORANGE','DARKORANGE','PINK','HOTPINK','MAROON','ORCHID1']
        
        
        if color_list:
           self.color_list = color_list
        else:
           self.color_list = self.color_list_default
        if color_label:       
           self.color_label = color_label
        else:
           self.color_label = self.color_list[0]

        self.selected_color_label = color_label
       
        self.SetBackgroundColour("grey90")
          
        self.InitDlg()   
        self.__ApplyLayout()      
    
    def InitDlg(self):
        self._btbox = DLGButtonPanel(self,style=wx.SUNKEN_BORDER)
      
        self.pnl_bcb = wx.Panel(self)   
        self.cdb_label_list = WX_CDB.getColourList()  
        self.cdb_info_list  = WX_CDB.getColourInfoList()  #

        bcb = BitmapComboBox(self.pnl_bcb,size=(140,-1))
        size_h_w = 16
        slidx=0
        for c in self.color_list:

            idx = self.cdb_label_list.index(c.upper())
            (r,g,b) = self.cdb_info_list[idx][1:]
            bmp = wx.EmptyBitmapRGBA(size_h_w, size_h_w,red=r,green=g,blue=b,alpha= wx.ALPHA_OPAQUE)
            bcb.Append(c, bmp, c)
            self.Bind(wx.EVT_COMBOBOX,self.OnCombo,bcb)
            if ( self.selected_color_label == c ):
               bcb.SetSelection( bcb.GetCount() -1)
           
    def __ApplyLayout(self):
 
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.pnl_bcb, 0, wx.EXPAND)
        vbox.Add(self._btbox,0,wx.EXPAND,border=5)
        self.SetSizerAndFit(vbox)
  
    
    def OnCombo(self, evt):
        bcb = evt.GetEventObject()
        idx = evt.GetInt()
        self.selected_color_label = bcb.GetString(idx)
       # cd  = bcb.GetClientData(idx)
        print "EVT_COMBOBOX: Id %d, string '%s'" % (idx, self.selected_color_label)
    
     
    def OnDestroy(self, event):
       print 'In OnDestroy'     
       event.Skip()
       return self.selected_color_label

    def OnSelect(self, e):
        
        i = e.GetString()
        print"OnSelect: "+ i
        print"\n"





          
"""          
#---------------------------------------------------------------------- 
class __ColorGridDialog(wx.Dialog):
   """"""
   def __init__(self,title="JuMEG TSV Groups Color Dialog",color_label=None,label_list=None): 
       super(ColorGridDialog,self).__init__(None,title=title) 
       cl=['BLACK','RED','GREEN','MAGENTA','BLUE','BROWN','CYAN','DARKGREEN','DARKORANGE','TURQUOISE','VIOLET','GREY50','GREY50','GREY50','GREY50']
       if label_list:
          self.label_list = label_list
       else:
          self.label_list = cl
       if color_label:       
          self.color_label = color_label
       else:
          self.color_label = self.label_list[0]
       
       self.SetBackgroundColour("grey90")
       #self.InitColorGrid()     
       self.InitDlg()   
       self.__ApplyLayout()      
      
     
   def InitDlg(self):
       self._btbox = DLGButtonPanel(self,style=wx.SUNKEN_BORDER)
       
       self.color_panel = wx.Panel(self)
       self.grid  = gridlib.Grid(self.color_panel)       
 
       sizer = wx.BoxSizer(wx.VERTICAL)
       sizer.Add(self.grid, 1, wx.EXPAND)
       self.color_panel.SetSizer(sizer)       
       
       cnt = len( self.label_list)
       self.grid.CreateGrid(cnt,2)
       self.grid.EnableEditing(False)
       self.Bind(gridlib.EVT_GRID_SELECT_CELL, self.OnSelectCell)
       self.Bind(gridlib.EVT_GRID_CELL_LEFT_CLICK, self.OnSelectCell)
     
       for i in range(cnt): 
           self.grid.SetCellValue(i,0,self.label_list[i])
           self.grid.SetCellBackgroundColour(i,1, self.label_list[i])
            
 
   def OnSelectCell(self,evt):
        
       row = None
       if evt.Selecting():
          row = evt.GetRow()
      
          print "Selected color : %d  -> %s\n" %(row,self.label_list[row])
        
          evt.Skip()
        
       if row:
          print row
          self.color_label = self.label_list[row]

   def OnClose(self, event):
       print 'In OnClose'
       event.Skip()

   def OnDestroy(self, event):
       print 'In OnDestroy'     
       event.Skip()
       return self.color_label

  #---
   def __ApplyLayout(self):
      
       vbox = wx.BoxSizer(wx.VERTICAL)
                  
       vbox.Add(self.color_panel,1,wx.ALL|wx.EXPAND,border=5)
       vbox.Add(self._btbox,0,wx.EXPAND,border=5)
       self.SetSizerAndFit(vbox)
  


class DLGColor(wx.Dialog):
     
      def __init__(self, parent, id, title,colors=None,color='grey90'):
          super(DLGColor,self).__init__(parent, id, title, size=(600,500), style=wx.DEFAULT_DIALOG_STYLE)
     
          cl=['BLACK','RED','GREEN','MAGENTA','BLUE','BROWN','CYAN','DARKGREEN','DARKORANGE','TURQUOISE','VIOLET','GREY50','GREY50','GREY50','GREY50']
        
          if not colors: 
             self.colors= cl
          
          #pnl = wx.Panel.__init__(self, parent, *args, **kwargs)
          self.SetBackgroundColour(color)
          vbox = wx.BoxSizer(wx.VERTICAL)
     
          self.lc = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
          self.lc.InsertColumn(0, 'Select Colour')
          #self.lc.InsertColumn(1, 'Capital')
          #self.lc.SetColumnWidth(0,100)
          #self.lc.SetColumnWidth(1,200)
          #self.lc.SetItemCount( len(self.colors) )
          # idx=0
          print len(self.colors)
         #cb=wx.ComboBox(self, -1, "default value",style=wx.DefaultSize,
         #                choices=self.colors)

          #self.lc.SetItemCount( len(self.colors))
          #for l in self.colors:
          #     num_items = self.lc.GetItemCount()
          #     print num_items
          #     print l
          #     self.lc.InsertStringItem(num_items,l)
          #     self.lc.SetItemBackgroundColour(num_items,l)     
               #self.lc.SetStringItem(num_items,1,l)
               #self.lc.SetItemBackgroundColour(num_items,'WHITE')   
               #self.lc.SetItemData(idx,l)    
               #idx +=1

          #self.Bind(wx.EVT_LIST_COL_CLICK, self.OnColClick, self.lc)
          
          vbox.Add(cb, 1, wx.EXPAND | wx.ALL, 3)
          self.SetSizer(vbox)
          
      def OnColClick(self, evt):
          obj = evt.GetEventObject()
          #item = self.lc.FindItemById(evt.GetId())
          #item = self.lc.FindItemById(evt.GetId())
          text = obj.GetText()
  
          print text
          
          # evt.Skip()
 """