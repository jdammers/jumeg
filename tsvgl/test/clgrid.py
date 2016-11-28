#!/usr/bin/env python
import wx
import wx.grid as gridlib
 

from jumeg.tsv.plot2d.jumeg_tsv_plot2d_data_info import JuMEG_TSV_PLOT2D_DATA_INFO

########################################################################
class MyForm(wx.Frame):
    """"""
 
    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, parent=None, title="A Simple Grid")
        panel = wx.Panel(self)
    
        self.info = JuMEG_TSV_PLOT2D_DATA_INFO()

        self.myGrid = gridlib.Grid(panel)
        
        self.init_grid()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.myGrid, 1, wx.EXPAND)
        panel.SetSizer(sizer)
 
    def init_grid(self):
        
        self.myGrid.CreateGrid(2, len( self.info.plt_color.label_list) )


if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()
