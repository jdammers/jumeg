#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 09:51:01 2018

@author: fboers
"""
import wx

__version__='2019.05.1403.001'

class JuMEG_wxAboutBox(object):
    def __init__(self):
        """    """
        self.name        = "JuMEG"
        self.version     = "0.007"
        self.description = "JuMEG MEG Data Analysis at INM4-MEG-FZJ"
        self.licence     = "Copyright, authors of JuMEG"
        self.copyright   = "(C) start - end author"
        self.website     = 'https://github.com/jdammers/jumeg'
        self.developer   = "Homer Simpson"
        self.docwriter   = None
        self.artist      = "JuMEG"
   #---     
    def show(self):
        if not self.docwriter:
           self.docwriter = self.developer
           
        info = wx.AboutDialogInfo()
        info.SetName(self.name)
        info.SetVersion(self.version)
        info.SetDescription(self.description)
        info.SetCopyright(self.copyright)
        info.SetWebSite(self.website)
        info.SetLicence(self.licence)
        info.AddDeveloper(self.developer)
        info.AddDocWriter(self.docwriter)
        info.AddArtist(self.artist)
        wx.AboutBox(info)