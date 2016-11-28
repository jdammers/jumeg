import wx
import numpy as np

class JuMEG_TSV_AXIS(wx.Panel):
    def __init__(self, parent, id=-1,height=30,num_range=(0,100,10),xticks=10):
        wx.Panel.__init__(self, parent, id, size=(-1,height), style=wx.SUNKEN_BORDER)
        
        self.parent = parent
        self.font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL,
            wx.FONTWEIGHT_NORMAL, False, 'Courier 10 Pitch')

        self.height   = height
        self.xticks   = xticks

        self.range_min=num_range[0]
        self.range_max=num_range[1]
        self.range_step=num_range[2]

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        
        self.xticks_range = np.arange(self.xticks)+1

    def OnPaint(self, event):
    
        #num = range(75, 700, 75)
        # self.num_range=range(self.range_min,self.range_max - self.range_step,self.range_step)
        xtick_vals = self.xticks_range * (self.range_max-self.range_min)/self.xticks_range.size  + self.range_min


        dc = wx.PaintDC(self)
        dc.SetFont(self.font)
        w, h = self.GetSize()

       # self.cw = self.parent.GetParent().cw

        #step = int(round(w / self.step))
        
        step = w / self.xticks_range.size
        j = 0

       # curs pos ?
       # till = (w / self.range_max ) * self.cw
       
       #  till = (w / 750.0) * self.cw
       # full = (w / 750.0) * 700


       # if self.cw >= 700:
       #     dc.SetPen(wx.Pen('#FFFFB8')) 
       #     dc.SetBrush(wx.Brush('#FFFFB8'))
       #     dc.DrawRectangle(0, 0, full, 30)
       #     dc.SetPen(wx.Pen('#ffafaf'))
       #     dc.SetBrush(wx.Brush('#ffafaf'))
       #     dc.DrawRectangle(full, 0, till-full, 30)
       # else: 
        dc.SetPen(wx.Pen('#FFFFB8'))
        dc.SetBrush(wx.Brush('#FFFFB8'))
       # dc.DrawRectangle(0, 0, till, self.height)
       # dc.DrawRectangle(0, 0, till, self.height)



        dc.SetPen(wx.Pen('#5C5142'))
        
      #  for i in range(step, 10*step, step):
        for i in (self.xticks_range):
               
            width, height = dc.GetTextExtent(str(xtick_vals[j]))
            xpos = i * step -width/2 
            dc.DrawLine(xpos, 0, xpos, 6)
            dc.DrawText(str(xtick_vals[j]), xpos, 8)
            j = j + 1
        
        print "DONE axis \n"

    def OnSize(self, event):
        self.Refresh()

"""
class Burning(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title, size=(330, 200))

        self.cw = 75

        panel = wx.Panel(self, -1)
        CenterPanel = wx.Panel(panel, -1)
        self.sld = wx.Slider(CenterPanel, -1, 75, 0, 750, (-1, -1), 
            (150, -1), wx.SL_LABELS)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)

        self.wid = Widget(panel, -1)
        hbox.Add(self.wid, 1, wx.EXPAND)

        hbox2.Add(CenterPanel, 1, wx.EXPAND)
        hbox3.Add(self.sld, 0, wx.TOP, 35)

        CenterPanel.SetSizer(hbox3)

        vbox.Add(hbox2, 1, wx.EXPAND)
        vbox.Add(hbox, 0, wx.EXPAND)


        self.Bind(wx.EVT_SCROLL, self.OnScroll)

        panel.SetSizer(vbox)

        self.sld.SetFocus()

        self.Centre()
        self.Show(True)

    def OnScroll(self, event):
        self.cw = self.sld.GetValue()
        self.wid.Refresh()


app = wx.App()
Burning(None, -1, 'Burning widget')
app.MainLoop()
"""
