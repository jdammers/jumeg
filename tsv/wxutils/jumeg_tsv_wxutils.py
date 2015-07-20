# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:05:40 2015

@author: fboers
"""

import wx

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
 