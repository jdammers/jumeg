# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:05:40 2015

@author: fboers
"""

import wx
import wx.lib.colourdb as WX_CDB
from wx.adv import BitmapComboBox

__version__ = "2019-09-13-001"

class DLGButtonPanel(wx.Panel):
      """Create the Button Panel.  CANCEL, Apply"""
      def __init__(self,parent,colour='grey90',buttons=None,AffirmativeId=None,*kargs,**kwargs):
          super().__init__(parent,*kargs,**kwargs)
          if not buttons:
             buttons = [wx.ID_CANCEL,wx.ID_APPLY]
             
          self.SetBackgroundColour(colour)
       
          hbox = wx.BoxSizer(wx.HORIZONTAL)
         
          for i in range(len(buttons)):
              hbox.Add(wx.Button(self,buttons[i]),0,wx.LEFT,border=2)
              if i < len( buttons ) -1:
                 hbox.Add((0,0),1,wx.LEFT,border=5)
         
          self.SetSizer(hbox)
          if AffirmativeId:
             self.GetParent().SetAffirmativeId(AffirmativeId)
          else:
             self.GetParent().SetAffirmativeId(buttons[-1])
             

########################################################################
class colourComboBoxDialog(wx.Dialog):
    """"""

   # import wx.lib.colourdb as WX_CDB
    
 
    #----------------------------------------------------------------------
    def __init__(self,title="JuMEG TSV Groups colour Dialog",colour_label=None,colour_list=None):
        """Constructor"""
       # wx.Dialog.__init__(self,title=title) 
        super(colourComboBoxDialog,self).__init__(None,title=title) 
        
    
        self.colour_list_default = ['BLACK','RED','AQUAMARINE','BLUE','MEDIUMBLUE','MIDNIGHTBLUE','ROYALBLUE','NAVYBLUE','CYAN',
                                   'GREEN','DARKGREEN','YELLOW','MAGENTA','VIOLET','PURPLE1','GREY40','GREY50','GREY60','GREY70',
                                   'GOLD','PERU','BROWN','ORANGE','DARKORANGE','PINK','HOTPINK','MAROON','ORCHID1']
        
        
        if colour_list:
           self.colour_list = colour_list
        else:
           self.colour_list = self.colour_list_default
        if colour_label:       
           self.colour_label = colour_label
        else:
           self.colour_label = self.colour_list[0]

        self.selected_colour_label = colour_label
       
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
        for c in self.colour_list:

            idx = self.cdb_label_list.index(c.upper())
            (r,g,b) = self.cdb_info_list[idx][1:]
            bmp = wx.EmptyBitmapRGBA(size_h_w, size_h_w,red=r,green=g,blue=b,alpha= wx.ALPHA_OPAQUE)
            bcb.Append(c, bmp, c)
            self.Bind(wx.EVT_COMBOBOX,self.OnCombo,bcb)
            if ( self.selected_colour_label == c ):
               bcb.SetSelection( bcb.GetCount() -1)
           
    def __ApplyLayout(self):
 
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.pnl_bcb, 0, wx.EXPAND)
        vbox.Add(self._btbox,0,wx.EXPAND,border=5)
        self.SetSizerAndFit(vbox)
  
    
    def OnCombo(self, evt):
        bcb = evt.GetEventObject()
        idx = evt.GetInt()
        self.selected_colour_label = bcb.GetString(idx)
       # cd  = bcb.GetClientData(idx)
        print( "EVT_COMBOBOX: Id %d, string '%s'" % (idx, self.selected_colour_label))
    
     
    def OnDestroy(self, event):
       print( 'In OnDestroy')
       event.Skip()
       return self.selected_colour_label

    def OnSelect(self, e):
        
        i = e.GetString()
        print("OnSelect: "+ i)
        print("\n")





          
"""          
#---------------------------------------------------------------------- 
class __colourGridDialog(wx.Dialog):
   """"""
   def __init__(self,title="JuMEG TSV Groups colour Dialog",colour_label=None,label_list=None): 
       super(colourGridDialog,self).__init__(None,title=title) 
       cl=['BLACK','RED','GREEN','MAGENTA','BLUE','BROWN','CYAN','DARKGREEN','DARKORANGE','TURQUOISE','VIOLET','GREY50','GREY50','GREY50','GREY50']
       if label_list:
          self.label_list = label_list
       else:
          self.label_list = cl
       if colour_label:       
          self.colour_label = colour_label
       else:
          self.colour_label = self.label_list[0]
       
       self.SetBackgroundColour("grey90")
       #self.InitcolourGrid()     
       self.InitDlg()   
       self.__ApplyLayout()      
      
     
   def InitDlg(self):
       self._btbox = DLGButtonPanel(self,style=wx.SUNKEN_BORDER)
       
       self.colour_panel = wx.Panel(self)
       self.grid  = gridlib.Grid(self.colour_panel)       
 
       sizer = wx.BoxSizer(wx.VERTICAL)
       sizer.Add(self.grid, 1, wx.EXPAND)
       self.colour_panel.SetSizer(sizer)       
       
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
      
          print "Selected colour : %d  -> %s\n" %(row,self.label_list[row])
        
          evt.Skip()
        
       if row:
          print row
          self.colour_label = self.label_list[row]

   def OnClose(self, event):
       print 'In OnClose'
       event.Skip()

   def OnDestroy(self, event):
       print 'In OnDestroy'     
       event.Skip()
       return self.colour_label

  #---
   def __ApplyLayout(self):
      
       vbox = wx.BoxSizer(wx.VERTICAL)
                  
       vbox.Add(self.colour_panel,1,wx.ALL|wx.EXPAND,border=5)
       vbox.Add(self._btbox,0,wx.EXPAND,border=5)
       self.SetSizerAndFit(vbox)
  


class DLGcolour(wx.Dialog):
     
      def __init__(self, parent, id, title,colours=None,colour='grey90'):
          super(DLGcolour,self).__init__(parent, id, title, size=(600,500), style=wx.DEFAULT_DIALOG_STYLE)
     
          cl=['BLACK','RED','GREEN','MAGENTA','BLUE','BROWN','CYAN','DARKGREEN','DARKORANGE','TURQUOISE','VIOLET','GREY50','GREY50','GREY50','GREY50']
        
          if not colours: 
             self.colours= cl
          
          #pnl = wx.Panel.__init__(self, parent, *args, **kwargs)
          self.SetBackgroundColour(colour)
          vbox = wx.BoxSizer(wx.VERTICAL)
     
          self.lc = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
          self.lc.InsertColumn(0, 'Select Colour')
          #self.lc.InsertColumn(1, 'Capital')
          #self.lc.SetColumnWidth(0,100)
          #self.lc.SetColumnWidth(1,200)
          #self.lc.SetItemCount( len(self.colours) )
          # idx=0
          print len(self.colours)
         #cb=wx.ComboBox(self, -1, "default value",style=wx.DefaultSize,
         #                choices=self.colours)

          #self.lc.SetItemCount( len(self.colours))
          #for l in self.colours:
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