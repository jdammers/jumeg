#!/usr/bin/env python

# The glassbrain class copied from The NeuroImaging Analysis Framework (NAF) repositories
# The code is covered under GNU GPL v2.

# Usage example.
'''
brain = ConnecBrain("fsaverage", "lh", "inflated")
coords = np.array([[-27., 23., 48.],
          [-41.,-60., 29.],
          [-64., -20., -9.],
          [ -7., 49., 18.],
          [ -7., -52., 26.]])
labels = ['MFG','AG','MTG','PCC','MPFC']
brain.add_coords(coords, color='green', labels=labels, scale_factor=1)
brain.add_arrow(coords[:2,:], color='red')
mlab.view(45,135)

# Note: If used in a jumeg module, use this below import statements.
# try:
#     import glassbrain
# except Exception as e:
#     print ('Unable to import glassbrain check mayavi and pysurfer config.')

'''

import numpy as np
from matplotlib.colors import colorConverter

from mayavi import mlab
from mayavi.mlab import pipeline as mp

import surfer

class ConnecBrain(surfer.Brain):
    """
    Subclass of sufer.Brain which allows adding co-ordinates and arrows
    to denote directional connectivity estimates
    """

    def __init__(self, subject_id, hemi, surf='inflated', curv=True,
                 title=None, config_opts={}, figure=None, subjects_dir=None,
                 views=['lat'], show_toolbar=False, offscreen=False,
                 opacity=0.3):

        # Call our main constructor
        surfer.Brain.__init__(self, subject_id, hemi, surf, views=views, curv=curv,
                              config_opts=config_opts, subjects_dir=subjects_dir)
        #surfer.Brain.__init__(self, subject_id, hemi, surf, curv, title,
        #                            config_opts, figure, subjects_dir,
        #                            views, show_toolbar, offscreen)

        # Initialise our arrows dictionary
        self.arrows_dict = dict()

        # Set all brain opacities
        for b in self._brain_list:
            b['brain']._geo_surf.actor.property.opacity = opacity

    def arrows(self):
        """Wrap to arrows"""
        return self._get_one_brain(self.arrows_dict, 'arrows')

    def add_coords(self, coords, map_surface=None, scale_factor=1.5,
                   color="red", alpha=1, name=None, labels=None, hemi=None,
                   text_size=5, txt_pos=[1.4, 1.1, 1.1]):
        """
        Plot locations onto the brain surface as spheres.

        :param coords: list of co-ordinates or (n, 3) numpy array.  Co-ordinate
            space must match that of the underlying MRI image
        :param map_surface: Freesurfer surf or None.
            surface to map coordinates through, or None to use raw coords
        :param scale_factor: int
            controls the size of the foci spheres
        :param color: matplotlib color code
            HTML name, RGB tuple or hex code
        :param alpha: float in [0, 1]
            opacity of coordinate spheres
        :param name: str
            internal name to use (_foci and _labels will be appended)
        :param labels:
            List of text strings used to label co-ordinates
        :param hemi: str | None
            If None, assumed to belong to the hemisphere being shown.
            If two hemispheresa are being shown, an error will be thrown
        :param text_size: int
            Text size of labels
        """

        hemi = self._check_hemi(hemi)

        if map_surface is None:
            foci_vtxs = surfer.utils.find_closest_vertices(self.geo[hemi].coords, coords)
            foci_coords = self.geo[hemi].coords[foci_vtxs]
        else:
            foci_surf = surfer.utils.Surface(self.subject_id, hemi, map_surface,
                                   subjects_dir=self.subjects_dir)
            foci_surf.load_geometry()
            foci_vtxs = surfer.utils.find_closest_vertices(foci_surf.coords, coords)
            foci_coords = self.geo[hemi].coords[foci_vtxs]

        # Convert the color code
        if not isinstance(color, tuple):
            color = colorConverter.to_rgb(color)

        if name is None:
            name = "coords_%s" % (max(len(self.foci_dict) + 1,
                                      len(self.labels_dict) + 1))

        views = self._toggle_render(False)

        # Store the coords in the foci list and the label in the labels list
        fl = []

        # Create the visualization
        for brain in self._brain_list:
            if brain['hemi'] == hemi:
                fl.append(mlab.points3d(foci_coords[:, 0],
                                        foci_coords[:, 1],
                                        foci_coords[:, 2],
                                        np.ones(foci_coords.shape[0]),
                                        scale_factor=(10. * scale_factor),
                                        color=color, opacity=alpha,
                                        name=name + '_foci',
                                        figure=brain['brain']._f))

        self.foci_dict[name + '_foci'] = fl

        if labels is not None:
            tl = []
            for i in range(coords.shape[0]):
                tl.append(mlab.text3d(foci_coords[i, 0]*txt_pos[0],
                                       foci_coords[i, 1]*txt_pos[1],
                                       foci_coords[i, 2]*txt_pos[2],
                                       labels[i],
                                       color=(1.0, 1.0, 1.0),
                                       scale=text_size,
                                       name=name + '_label',
                                       figure=brain['brain']._f))

            self.labels_dict[name + '_label'] = fl

        self._toggle_render(True, views)

    def add_arrow(self, coords, map_surface=None, tube_radius=3.0,
                  color="white", alpha=1, name=None, hemi=None):
        """
        Add an arrow across the brain between two co-ordinates

        :param coords: list of co-ordinates or (n, 3) numpy array.  Co-ordinate
            space must match that of the underlying MRI image
        :param tube_radius: float
            controls the size of the arrow
        :param color: matplotlib color code
            HTML name, RGB tuple or hex code
        :param alpha: float in [0, 1]
            opacity of coordinate spheres
        :param name: str
            internal name to use
        :param hemi: str | None
            If None, assumed to belong to the hemisphere being shown.
            If two hemispheresa are being shown, an error will be thrown
        """

        hemi = self._check_hemi(hemi)

        if map_surface is None:
            foci_vtxs = surfer.utils.find_closest_vertices(self.geo[hemi].coords, coords)
            foci_coords = self.geo[hemi].coords[foci_vtxs]
        else:
            foci_surf = surfer.utils.Surface(self.subject_id, hemi, map_surface,
                                   subjects_dir=self.subjects_dir)
            foci_surf.load_geometry()
            foci_vtxs = surfer.utils.find_closest_vertices(foci_surf.coords, coords)
            foci_coords = self.geo[hemi].coords[foci_vtxs]

        # foci_vtxs = surfer.utils.find_closest_vertices(self.geo[hemi].coords, coords)
        # foci_coords = self.geo[hemi].coords[foci_vtxs]

        # Convert the color code
        if not isinstance(color, tuple):
            color = colorConverter.to_rgb(color)

        if name is None:
            name = "arrow_%s" % (len(self.arrows_dict) + 1)

        nsegs = 100

        x = np.linspace(foci_coords[0, 0], foci_coords[1, 0], nsegs)
        y = np.linspace(foci_coords[0, 1], foci_coords[1, 1], nsegs)
        z = np.linspace(foci_coords[0, 2], foci_coords[1, 2], nsegs)

        line_coords = np.vstack((x, y, z)).transpose()
        step = 5
        idx_a = list(range(0, nsegs+1, step))
        idx_b = list(range(10, nsegs+1, step))

        views = self._toggle_render(False)

        al = []

        for brain in self._brain_list:
            if brain['hemi'] == hemi:
                for start,end in zip(idx_a, idx_b):

                    seg_width = tube_radius - (start*(tube_radius-.5)/100.)

                    al.append(mlab.plot3d(line_coords[start:end, 0],
                                line_coords[start:end, 1],
                                line_coords[start:end, 2],
                                np.ones_like(line_coords[start:end, 0]),
                                color=color, opacity=alpha,
                                tube_radius=seg_width,
                                name=name,
                                figure=brain['brain']._f))

        self.arrows_dict[name] = al

        self._toggle_render(True, views)
